# -*- coding: utf-8 -*-
"""Depth protrusion detection: find handle as geometry sticking out from door plane."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .door_roi import DoorRoi
from .handle_geometry import (
    contour_centroid,
    extrapolate_grasp_horizontal,
    grasp_from_lever_mask,
    split_backplate_lever,
)
from .types import HandleDetection


def _valid_depth_mask(depth_mm: np.ndarray, roi: DoorRoi) -> np.ndarray:
    sy, sx = roi.as_slice()
    patch = depth_mm[sy, sx]
    valid = (patch > 250) & (patch < 4000)
    out = np.zeros(depth_mm.shape, dtype=np.uint8)
    out[sy, sx] = (valid.astype(np.uint8) * 255)
    return out


def protrusion_mask_in_roi(
    depth_mm: np.ndarray,
    roi: DoorRoi,
    *,
    min_mm: float = 12.0,
    max_mm: float = 85.0,
    bg_percentile: float = 72.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build protrusion mask inside ROI.

    Returns (full-size protrusion mask 0/255, full-size valid depth mask 0/255).
    Protrusion = background_depth - pixel_depth (mm); positive means closer to camera.
    """
    img_h, img_w = depth_mm.shape[:2]
    sy, sx = roi.as_slice()
    patch = depth_mm[sy, sx].astype(np.float32)
    valid = (patch > 250) & (patch < 4000)
    protrusion = np.zeros((img_h, img_w), dtype=np.uint8)
    valid_full = np.zeros((img_h, img_w), dtype=np.uint8)

    if int(np.count_nonzero(valid)) < 80:
        return protrusion, valid_full

    bg = float(np.percentile(patch[valid], bg_percentile))
    local = np.zeros_like(patch, dtype=np.float32)
    local[valid] = bg - patch[valid]
    local_mask = ((local > min_mm) & (local < max_mm)).astype(np.uint8) * 255
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    protrusion[sy, sx] = local_mask
    valid_full[sy, sx] = (valid.astype(np.uint8) * 255)
    return protrusion, valid_full


def _dark_handle_ratio(bgr: np.ndarray, cnt: np.ndarray) -> float:
    """Black/dark lever on light door: wall blobs usually lack dark pixels."""
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray[ys, xs] < 95))


def _score_handle_blob(
    cnt: np.ndarray,
    roi: DoorRoi,
    img_h: int,
    img_w: int,
    *,
    bgr: Optional[np.ndarray] = None,
) -> float:
    area = float(cv2.contourArea(cnt))
    if area < 80:
        return 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 12 or h < 10:
        return 0.0
    aspect = w / max(float(h), 1.0)
    if aspect < 0.5 or aspect > 8.0:
        return 0.0
    if w > roi.w * 0.95 and h > roi.h * 0.95:
        return 0.0

    cx = x + w * 0.5
    cy = y + h * 0.5
    in_roi = roi.x <= cx <= roi.x + roi.w and roi.y <= cy <= roi.y + roi.h
    if not in_roi:
        return 0.0

    horizontal_bonus = 1.0 if 1.0 <= aspect <= 5.0 else 0.65
    size_bonus = min(1.0, area / 1500.0)
    score = area * horizontal_bonus * size_bonus

    if bgr is not None:
        dark_ratio = _dark_handle_ratio(bgr, cnt)
        if dark_ratio < 0.04:
            score *= 0.15
        else:
            score *= 0.55 + 0.45 * min(1.0, dark_ratio / 0.35)

    return score


def detect_in_roi(
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    roi: DoorRoi,
    *,
    extrapolate: bool = False,
) -> HandleDetection:
    """Detect handle inside one ROI using depth protrusion."""
    img_h, img_w = bgr.shape[:2]
    roi = roi.clamp(img_h, img_w)

    protrusion, valid_depth = protrusion_mask_in_roi(depth_mm, roi)
    contours, _ = cv2.findContours(protrusion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt: Optional[np.ndarray] = None
    best_score = 0.0
    for cnt in contours:
        score = _score_handle_blob(cnt, roi, img_h, img_w, bgr=bgr)
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return HandleDetection(
            success=False,
            message="no protrusion blob",
            debug={
                "roi": roi,
                "protrusion_mask": protrusion,
                "valid_depth_mask": valid_depth,
            },
        )

    handle_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(handle_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
    handle_contour = best_cnt

    _, lever_mask, backplate_cnt, lever_cnt = split_backplate_lever(handle_mask, handle_contour)
    backplate_center = contour_centroid(backplate_cnt)

    grasp_point, lever_axis = grasp_from_lever_mask(lever_mask, backplate_cnt, img_h, img_w)
    if grasp_point is None:
        return HandleDetection(
            success=False,
            message="grasp invalid in roi",
            handle_contour=handle_contour,
            backplate_contour=backplate_cnt,
            lever_contour=lever_cnt,
            backplate_center=backplate_center,
            debug={"roi": roi, "protrusion_mask": protrusion, "handle_mask": handle_mask},
        )

    if extrapolate:
        grasp_point = extrapolate_grasp_horizontal(grasp_point, handle_mask, backplate_cnt, img_w)

    bx, by, bw, bh = cv2.boundingRect(backplate_cnt)
    lever_span = float(cv2.boundingRect(lever_cnt)[2])
    span_ratio = lever_span / max(float(max(bw, 8)), 1.0)
    confidence = float(min(1.0, 0.35 * min(best_score / 2500.0, 1.0) + 0.65 * min(span_ratio / 1.6, 1.0)))

    return HandleDetection(
        success=True,
        grasp_point=grasp_point,
        handle_contour=handle_contour,
        backplate_contour=backplate_cnt,
        lever_contour=lever_cnt,
        lever_axis=lever_axis,
        backplate_center=backplate_center,
        confidence=confidence,
        message="depth_protrusion",
        debug={
            "roi": roi,
            "protrusion_mask": protrusion,
            "valid_depth_mask": valid_depth,
            "handle_mask": handle_mask,
            "lever_mask": lever_mask,
            "blob_score": best_score,
            "dark_ratio": _dark_handle_ratio(bgr, best_cnt),
        },
    )


def detect_depth_protrusion(
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    rois: list[DoorRoi],
    *,
    extrapolate: bool = False,
) -> HandleDetection:
    """Run protrusion detection across ROIs and return the best result."""
    if depth_mm is None or depth_mm.size == 0:
        return HandleDetection(success=False, message="no depth")
    if depth_mm.shape[:2] != bgr.shape[:2]:
        return HandleDetection(success=False, message="depth rgb size mismatch")

    best: Optional[HandleDetection] = None
    best_rank = -1.0
    for roi in rois:
        det = detect_in_roi(bgr, depth_mm, roi, extrapolate=extrapolate)
        rank = det.confidence if det.success else -1.0
        if det.success:
            rank *= 0.5 + 0.5 * roi.confidence
        if rank > best_rank:
            best_rank = rank
            best = det

    if best is not None and best.success:
        best.debug["roi_mode"] = rois[0].source if rois else ""
        return best

    return HandleDetection(
        success=False,
        message="depth protrusion failed in all rois",
        debug={"roi_count": len(rois)},
    )
