"""Tests for semantic-segmentation aggregate metrics."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray

from lightly_studio.evaluation import segmentation_aggregate
from lightly_studio.models.evaluation_metrics import SegmentationMetrics

CAT = uuid4()
DOG = uuid4()
LABEL_NAMES = {CAT: "cat", DOG: "dog"}
SHAPE = (4, 4)


def _mask(rows: slice) -> NDArray[np.bool_]:
    mask = np.zeros(SHAPE, dtype=np.bool_)
    mask[rows, :] = True
    return mask


def _iou_by_label(result: SegmentationMetrics) -> dict[str, float]:
    return {entry.label: entry.iou for entry in result.per_class}


def test_perfect_segmentation() -> None:
    masks = {CAT: _mask(slice(0, 2))}  # top half

    result = segmentation_aggregate.compute(
        gt_masks_per_image=[masks],
        pred_masks_per_image=[dict(masks)],
        image_shapes=[SHAPE],
        label_names=LABEL_NAMES,
    )

    assert _iou_by_label(result) == {"cat": pytest.approx(1.0)}
    assert result.mean_iou == pytest.approx(1.0)
    # Foreground and background both agree everywhere.
    assert result.pixel_accuracy == pytest.approx(1.0)


def test_partial_overlap() -> None:
    gt = {CAT: _mask(slice(0, 2))}  # top two rows, 8 px
    pred = {CAT: _mask(slice(0, 1))}  # top row, 4 px (subset)

    result = segmentation_aggregate.compute(
        gt_masks_per_image=[gt],
        pred_masks_per_image=[pred],
        image_shapes=[SHAPE],
        label_names=LABEL_NAMES,
    )

    assert _iou_by_label(result) == {"cat": pytest.approx(0.5)}  # 4 / 8
    # Row 0 agrees (cat), row 1 wrong, rows 2-3 agree (background) -> 12 / 16.
    assert result.pixel_accuracy == pytest.approx(0.75)


def test_mean_over_classes() -> None:
    gt = {CAT: _mask(slice(0, 2)), DOG: _mask(slice(2, 4))}
    pred = {CAT: _mask(slice(0, 2)), DOG: _mask(slice(2, 3))}  # dog half right

    result = segmentation_aggregate.compute(
        gt_masks_per_image=[gt],
        pred_masks_per_image=[pred],
        image_shapes=[SHAPE],
        label_names=LABEL_NAMES,
    )

    assert _iou_by_label(result) == {"cat": pytest.approx(1.0), "dog": pytest.approx(0.5)}
    assert result.mean_iou == pytest.approx(0.75)


def test_class_only_in_prediction_has_iou_zero() -> None:
    result = segmentation_aggregate.compute(
        gt_masks_per_image=[{CAT: _mask(slice(0, 2))}],
        pred_masks_per_image=[{CAT: _mask(slice(0, 2)), DOG: _mask(slice(2, 4))}],
        image_shapes=[SHAPE],
        label_names=LABEL_NAMES,
    )

    assert _iou_by_label(result) == {"cat": pytest.approx(1.0), "dog": pytest.approx(0.0)}
    assert result.mean_iou == pytest.approx(0.5)


def test_pools_intersection_and_union_over_images() -> None:
    # Two images: image 1 perfect, image 2 disjoint. Pooled IoU = 8 / 24.
    result = segmentation_aggregate.compute(
        gt_masks_per_image=[{CAT: _mask(slice(0, 2))}, {CAT: _mask(slice(0, 2))}],
        pred_masks_per_image=[{CAT: _mask(slice(0, 2))}, {CAT: _mask(slice(2, 4))}],
        image_shapes=[SHAPE, SHAPE],
        label_names=LABEL_NAMES,
    )

    # Image 1: I=8, U=8. Image 2: I=0, U=16. Pooled: 8 / 24.
    assert _iou_by_label(result) == {"cat": pytest.approx(8 / 24)}


def test_empty() -> None:
    result = segmentation_aggregate.compute(
        gt_masks_per_image=[],
        pred_masks_per_image=[],
        image_shapes=[],
        label_names=LABEL_NAMES,
    )

    assert result.per_class == []
    assert result.mean_iou == pytest.approx(0.0)
    assert result.pixel_accuracy == pytest.approx(0.0)
