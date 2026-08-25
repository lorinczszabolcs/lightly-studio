"""Aggregate semantic-segmentation metrics pooled over all images.

Dataset intersection over union pools the per-class intersection and union over
every image, rather than averaging per-image values, so large and small images
contribute in proportion to their pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from lightly_studio.models.evaluation_metrics import ClassIoU, SegmentationMetrics


def compute(
    gt_masks_per_image: Sequence[Mapping[UUID, NDArray[np.bool_]]],
    pred_masks_per_image: Sequence[Mapping[UUID, NDArray[np.bool_]]],
    image_shapes: Sequence[tuple[int, int]],
    label_names: dict[UUID, str],
) -> SegmentationMetrics:
    """Compute per-class IoU, mean IoU, and pixel accuracy pooled over all images.

    A class is scored when it has any pixel in the ground truth or the prediction
    of some image. Pixel accuracy counts pixels whose class matches, including
    pixels that are background in both.

    Args:
        gt_masks_per_image: Ground-truth per-class binary masks, one map per image.
        pred_masks_per_image: Prediction per-class binary masks, in the same image order.
        image_shapes: ``(height, width)`` of each image, in the same order.
        label_names: Mapping from label ID to label name.

    Returns:
        The pooled segmentation metrics.
    """
    intersection: dict[UUID, int] = {}
    union: dict[UUID, int] = {}
    correct_pixels = 0
    total_pixels = 0

    for gt_masks, pred_masks, (height, width) in zip(
        gt_masks_per_image, pred_masks_per_image, image_shapes
    ):
        total_pixels += height * width
        empty = np.zeros((height, width), dtype=np.bool_)
        gt_foreground = empty.copy()
        pred_foreground = empty.copy()
        for label in set(gt_masks) | set(pred_masks):
            gt_mask = gt_masks.get(label, empty)
            pred_mask = pred_masks.get(label, empty)
            overlap = int(np.logical_and(gt_mask, pred_mask).sum())
            intersection[label] = intersection.get(label, 0) + overlap
            union[label] = union.get(label, 0) + int(np.logical_or(gt_mask, pred_mask).sum())
            correct_pixels += overlap
            gt_foreground |= gt_mask
            pred_foreground |= pred_mask
        # Pixels that are background in both ground truth and prediction also match.
        correct_pixels += int(np.logical_and(~gt_foreground, ~pred_foreground).sum())

    scored_labels = sorted(
        (label for label, pixels in union.items() if pixels > 0),
        key=lambda label: label_names[label],
    )
    per_class = [
        ClassIoU(label=label_names[label], iou=intersection[label] / union[label])
        for label in scored_labels
    ]
    mean_iou = float(np.mean([entry.iou for entry in per_class])) if per_class else 0.0
    pixel_accuracy = correct_pixels / total_pixels if total_pixels else 0.0
    return SegmentationMetrics(
        per_class=per_class,
        mean_iou=mean_iou,
        pixel_accuracy=pixel_accuracy,
    )
