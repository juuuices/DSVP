"""Scene-aware EA-UAV datasets for spatial and causal stream training."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ultralytics.utils import TQDM, colorstr
from ultralytics.utils.instance import Instances

from .augment import Compose, Format, LetterBox, v8_transforms
from .base import BaseDataset
from .build import build_dataloader
from .utils import IMG_FORMATS


def _labelme_header(path: Path) -> dict:
    """Read LabelMe metadata without loading the large embedded base64 image."""
    marker = b'"imageData"'
    data = bytearray()
    with path.open("rb") as file:
        while marker not in data:
            chunk = file.read(8192)
            if not chunk:
                break
            data.extend(chunk)
    if marker in data:
        data = data.split(marker, 1)[0].rstrip()
        if data.endswith(b","):
            data = data[:-1]
        data.extend(b"}")
    return json.loads(data.decode("utf-8"))


def _scene_and_timestamp(path: Path):
    scene_id = path.parent.name
    try:
        timestamp = int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        timestamp = 0
    return scene_id, timestamp


def _sibling_modality(image_path: Path, directory: str, suffix: str) -> Path:
    """Map .../<split>/images/<scene>/<frame> to another modality folder."""
    parts = list(image_path.parts)
    try:
        image_index = parts.index("images")
    except ValueError as error:
        raise ValueError(f"Expected an 'images' directory in {image_path}") from error
    parts[image_index] = directory
    return Path(*parts).with_suffix(suffix)


class SceneDetectionDataset(BaseDataset):
    """Read scene-grouped RGB frames and LabelMe boxes in causal order."""

    def __init__(self, *args, data=None, task="detect", stream=False, **kwargs):
        if task != "detect":
            raise ValueError("SceneDetectionDataset supports only object detection.")
        self.data = data
        self.stream = stream
        self.use_segments = self.use_keypoints = self.use_obb = False
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path):
        paths = img_path if isinstance(img_path, list) else [img_path]
        files = []
        for root in paths:
            root = Path(root)
            if not root.is_dir():
                raise FileNotFoundError(f"Scene image directory does not exist: {root}")
            files.extend(
                str(path)
                for path in root.rglob("*.*")
                if path.suffix[1:].lower() in IMG_FORMATS
            )
        files.sort(key=lambda item: (_scene_and_timestamp(Path(item))[0], _scene_and_timestamp(Path(item))[1]))
        if not files:
            raise FileNotFoundError(f"No images found below {img_path}")
        return files

    def get_labels(self):
        labels = []
        previous_scene = None
        progress = TQDM(self.im_files, desc=f"{self.prefix}Reading scene labels", total=len(self.im_files))
        for image_name in progress:
            image_path = Path(image_name)
            scene_id, timestamp = _scene_and_timestamp(image_path)
            label_path = _sibling_modality(image_path, "labels", ".json")
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing LabelMe annotation: {label_path}")

            metadata = _labelme_header(label_path)
            with Image.open(image_path) as image:
                width, height = image.size
            boxes = []
            classes = []
            for shape in metadata.get("shapes", []):
                if shape.get("label") != "UAV" or len(shape.get("points", [])) < 2:
                    continue
                (x1, y1), (x2, y2) = shape["points"][:2]
                x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
                y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append(
                    (
                        ((x1 + x2) / 2) / width,
                        ((y1 + y2) / 2) / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    )
                )
                classes.append((0.0,))

            labels.append(
                {
                    "im_file": str(image_path),
                    "shape": (height, width),
                    "cls": np.asarray(classes, dtype=np.float32).reshape(-1, 1),
                    "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "scene_id": scene_id,
                    "timestamp": timestamp,
                    "new_sequence": scene_id != previous_scene,
                }
            )
            previous_scene = scene_id
        return labels

    def build_transforms(self, hyp=None):
        if self.augment and not self.stream:
            hyp.mosaic = hyp.mosaic if not self.rect else 0.0
            hyp.mixup = hyp.mixup if not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            if self.stream:
                height, width = self.labels[0]["shape"]
                ratio = self.imgsz / max(height, width)
                stream_shape = (
                    math.ceil(height * ratio / self.stride + 0.5) * self.stride,
                    math.ceil(width * ratio / self.stride + 0.5) * self.stride,
                )
                transforms = Compose([LetterBox(new_shape=stream_shape, scaleup=False)])
            else:
                transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=False,
                bgr=hyp.bgr if self.augment and not self.stream else 0.0,
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        hyp.mosaic = hyp.copy_paste = hyp.mixup = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        bboxes = label.pop("bboxes")
        label.pop("segments", None)
        label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        label["instances"] = Instances(
            bboxes,
            np.zeros((0, 1000, 2), dtype=np.float32),
            None,
            bbox_format=bbox_format,
            normalized=normalized,
        )
        return label

    @staticmethod
    def collate_fn(batch):
        output = {}
        keys = batch[0].keys()
        values = list(zip(*[list(item.values()) for item in batch]))
        for index, key in enumerate(keys):
            value = values[index]
            if key == "img":
                import torch

                value = torch.stack(value, 0)
            elif key in {"bboxes", "cls"}:
                import torch

                value = torch.cat(value, 0)
            output[key] = value
        output["batch_idx"] = list(output["batch_idx"])
        for index in range(len(output["batch_idx"])):
            output["batch_idx"][index] += index
        import torch

        output["batch_idx"] = torch.cat(output["batch_idx"], 0)
        return output


def build_scene_dataset(cfg, img_path, batch, data, mode="train", stride=32, stream=False):
    """Build a standard spatial or ordered stream dataset from scene folders."""
    return SceneDetectionDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=1 if stream else batch,
        augment=mode == "train" and not stream,
        hyp=cfg,
        rect=mode == "val" and not stream,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.5 if mode == "val" else 0.0,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        stream=stream,
    )


def build_stream_dataloader(dataset, workers=0):
    """Return a strictly ordered batch-1 dataloader."""
    return build_dataloader(dataset, batch=1, workers=workers, shuffle=False, rank=-1)
