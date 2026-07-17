"""YOLO-format bounding-box dataset."""

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM
from ultralytics.utils.instance import Instances
from .augment import Compose, Format, LetterBox, v8_transforms
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image_label,
)

DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """Load detection images and normalized `class x y w h` labels."""

    def __init__(self, *args, data=None, task="detect", **kwargs):
        if task != "detect":
            raise ValueError("This minimal dataset loader supports only object detection.")
        self.data = data
        self.use_segments = False
        self.use_keypoints = False
        self.use_obb = False
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        cache = {"labels": []}
        missing = found = empty = corrupt = 0
        messages = []
        description = f"{self.prefix}Scanning {path.parent / path.stem}..."
        with ThreadPool(NUM_THREADS) as pool:
            inputs = zip(
                self.im_files,
                self.label_files,
                repeat(self.prefix),
                repeat(False),
                repeat(len(self.data["names"])),
                repeat(0),
                repeat(0),
            )
            progress = TQDM(pool.imap(verify_image_label, inputs), desc=description, total=len(self.im_files))
            for im_file, labels, shape, segments, _, nm, nf, ne, nc, message in progress:
                missing += nm
                found += nf
                empty += ne
                corrupt += nc
                if im_file:
                    cache["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": labels[:, 0:1],
                            "bboxes": labels[:, 1:],
                            "segments": segments,
                            "keypoints": None,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if message:
                    messages.append(message)
                progress.desc = (
                    f"{description} {found} images, {missing + empty} backgrounds, {corrupt} corrupt"
                )
            progress.close()

        if messages:
            LOGGER.info("\n".join(messages))
        if found == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        cache["hash"] = get_hash(self.label_files + self.im_files)
        cache["results"] = found, missing, empty, corrupt, len(self.im_files)
        cache["msgs"] = messages
        save_dataset_cache_file(self.prefix, path, cache, DATASET_CACHE_VERSION)
        return cache

    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False

        found, missing, empty, corrupt, total = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            description = (
                f"{self.prefix}Scanning {cache_path}... "
                f"{found} images, {missing + empty} backgrounds, {corrupt} corrupt"
            )
            TQDM(None, desc=description, total=total, initial=total)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        for key in ("hash", "version", "msgs"):
            cache.pop(key)
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"No images found in {cache_path}. {HELP_URL}")
        self.im_files = [label["im_file"] for label in labels]
        if sum(len(label["cls"]) for label in labels) == 0:
            LOGGER.warning(f"No labels found in {cache_path}. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if not self.rect else 0.0
            hyp.mixup = hyp.mixup if not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
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
                bgr=hyp.bgr if self.augment else 0.0,
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
        empty_segments = np.zeros((0, 1000, 2), dtype=np.float32)
        label["instances"] = Instances(
            bboxes, empty_segments, None, bbox_format=bbox_format, normalized=normalized
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
                value = torch.stack(value, 0)
            if key in {"bboxes", "cls"}:
                value = torch.cat(value, 0)
            output[key] = value
        output["batch_idx"] = list(output["batch_idx"])
        for index in range(len(output["batch_idx"])):
            output["batch_idx"][index] += index
        output["batch_idx"] = torch.cat(output["batch_idx"], 0)
        return output
