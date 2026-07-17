"""DSVP detection loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors
from .metrics import bbox_iou


class DFLoss(nn.Module):
    """Distribution focal loss."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        left = target.long()
        right = left + 1
        weight_left = right - target
        weight_right = 1 - weight_left
        return (
            F.cross_entropy(pred_dist, left.view(-1), reduction="none").view(left.shape) * weight_left
            + F.cross_entropy(pred_dist, right.view(-1), reduction="none").view(left.shape) * weight_right
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """IoU and distribution focal losses."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        iou_loss = ((1.0 - iou) * weight).sum() / target_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            dfl_loss = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * weight
            dfl_loss = dfl_loss.sum() / target_sum
        else:
            dfl_loss = torch.tensor(0.0, device=pred_dist.device)
        return iou_loss, dfl_loss


class v8DetectionLoss:
    """Box, class, and DFL losses for DSVP."""

    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.hyp = model.args
        head = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.stride = head.stride
        self.nc = head.nc
        self.reg_max = head.reg_max
        self.no = self.nc + self.reg_max * 4
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        rows, columns = targets.shape
        if rows == 0:
            return torch.zeros(batch_size, 0, columns - 1, device=self.device)
        image_indices = targets[:, 0]
        _, counts = image_indices.unique(return_counts=True)
        output = torch.zeros(batch_size, counts.max(), columns - 1, device=self.device)
        for index in range(batch_size):
            matches = image_indices == index
            count = matches.sum()
            if count:
                output[index, :count] = targets[matches, 1:]
        output[..., 1:5] = xywh2xyxy(output[..., 1:5].mul_(scale_tensor))
        return output

    def bbox_decode(self, anchor_points, pred_dist):
        batch, anchors, channels = pred_dist.shape
        pred_dist = (
            pred_dist.view(batch, anchors, 4, channels // 4)
            .softmax(3)
            .matmul(self.proj.to(dtype=pred_dist.dtype))
        )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        losses = torch.zeros(3, device=self.device)
        features = preds[1] if isinstance(preds, tuple) else preds
        pred_dist, pred_scores = torch.cat(
            [feature.view(features[0].shape[0], self.no, -1) for feature in features], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        image_size = torch.tensor(features[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(features, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, image_size[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_dist)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_sum = max(target_scores.sum(), 1)
        losses[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_sum
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            losses[0], losses[2] = self.bbox_loss(
                pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_sum, fg_mask
            )
        losses[0] *= self.hyp.box
        losses[1] *= self.hyp.cls
        losses[2] *= self.hyp.dfl
        return losses.sum() * batch_size, losses.detach()
