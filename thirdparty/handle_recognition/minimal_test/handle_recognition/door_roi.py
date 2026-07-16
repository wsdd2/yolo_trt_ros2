# -*- coding: utf-8 -*-
"""Door frame ROI and handle search region selection (RGB + optional depth)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class DoorRoi:
    """A rectangular search region for handle detection."""

    x: int
    y: int
    w: int
    h: int
    source: str
    confidence: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_slice(self) -> tuple[slice, slice]:
        return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)

    def clamp(self, img_h: int, img_w: int) -> "DoorRoi":
        x0 = max(0, min(self.x, img_w - 1))
        y0 = max(0, min(self.y, img_h - 1))
        x1 = max(x0 + 1, min(self.x + self.w, img_w))
        y1 = max(y0 + 1, min(self.y + self.h, img_h))
        return DoorRoi(x0, y0, x1 - x0, y1 - y0, self.source, self.confidence, dict(self.meta))


def _find_door_bbox(bgr: np.ndarray) -> tuple[Optional[tuple[int, int, int, int]], float]:
    """Detect a tall door-like rectangle from edges and contours."""
    img_h, img_w = bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[tuple[int, int, int, int]] = None
    best_score = 0.0

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < img_h * img_w * 0.08:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h < img_h * 0.35 or w > img_w * 0.65:
            continue
        aspect = h / max(float(w), 1.0)
        if aspect < 1.8:
            continue
        rectangularity = area / float(w * h)
        if rectangularity < 0.25:
            continue
        score = area * rectangularity * min(aspect / 3.0, 1.5)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=int(img_h * 0.25), maxLineGap=20)
        if lines is not None:
            vertical_x: list[int] = []
            for line in lines[:, 0]:
                x1, y1, x2, y2 = map(int, line)
                if abs(x2 - x1) < 15 and abs(y2 - y1) > img_h * 0.3:
                    vertical_x.extend([x1, x2])
            if len(vertical_x) >= 2:
                x0, x1 = min(vertical_x), max(vertical_x)
                w = x1 - x0
                if w > img_w * 0.08 and w < img_w * 0.7:
                    best = (max(0, x0 - 10), int(img_h * 0.05), min(img_w, w + 20), int(img_h * 0.9))
                    best_score = float(w * img_h * 0.3)

    if best is None:
        return None, 0.0
    conf = float(min(1.0, best_score / (img_h * img_w * 0.35)))
    return best, conf


def find_door_stile_rois(bgr: np.ndarray) -> list[DoorRoi]:
    """If a full door frame is visible, return left/right stile ROIs."""
    img_h, img_w = bgr.shape[:2]
    bbox, conf = _find_door_bbox(bgr)
    if bbox is None or conf < 0.15:
        return []

    x, y, w, h = bbox
    stile_w = max(40, int(w * 0.22))
    y0 = y + int(h * 0.22)
    y1 = y + int(h * 0.78)
    roi_h = max(60, y1 - y0)

    rois = [
        DoorRoi(x, y0, stile_w, roi_h, "door_stile_left", conf, {"door_bbox": bbox, "side": "left"}),
        DoorRoi(x + w - stile_w, y0, stile_w, roi_h, "door_stile_right", conf, {"door_bbox": bbox, "side": "right"}),
    ]
    return [r.clamp(img_h, img_w) for r in rois]


def _protrusion_hint_rois(depth_mm: np.ndarray, img_h: int, img_w: int) -> list[DoorRoi]:
    """Coarse protrusion-based ROIs when door frame is not found."""
    valid = (depth_mm > 300) & (depth_mm < 3500)
    if int(np.count_nonzero(valid)) < 500:
        return []

    bg = float(np.percentile(depth_mm[valid], 70))
    protrusion = np.zeros_like(depth_mm, dtype=np.float32)
    protrusion[valid] = bg - depth_mm[valid].astype(np.float32)
    mask = ((protrusion > 15) & (protrusion < 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois: list[DoorRoi] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 120 or area > img_h * img_w * 0.08:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(float(h), 1.0)
        if aspect < 0.8 or aspect > 6.0:
            continue
        pad = 20
        rois.append(
            DoorRoi(
                max(0, x - pad),
                max(0, y - pad),
                min(img_w, w + 2 * pad),
                min(img_h, h + 2 * pad),
                "depth_protrusion_hint",
                float(min(1.0, area / 2000.0)),
                {"area": float(area), "aspect": aspect},
            )
        )
    rois.sort(key=lambda r: r.confidence, reverse=True)
    return [r.clamp(img_h, img_w) for r in rois[:5]]


def _rgb_edge_hint_rois(bgr: np.ndarray) -> list[DoorRoi]:
    """Compact high-contrast blobs that may be handles (material-agnostic)."""
    img_h, img_w = bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois: list[DoorRoi] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150 or area > img_h * img_w * 0.04:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or h < 12:
            continue
        aspect = w / max(float(h), 1.0)
        if aspect < 0.6 or aspect > 7.0:
            continue
        score = area * (1.0 if 1.2 <= aspect <= 4.5 else 0.6)
        pad = 25
        rois.append(
            DoorRoi(
                max(0, x - pad),
                max(0, y - pad),
                min(img_w, w + 2 * pad),
                min(img_h, h + 2 * pad),
                "rgb_edge_hint",
                float(min(1.0, score / 3000.0)),
                {"area": float(area), "aspect": aspect},
            )
        )
    rois.sort(key=lambda r: r.confidence, reverse=True)
    return [r.clamp(img_h, img_w) for r in rois[:5]]


def find_handle_search_rois(
    bgr: np.ndarray,
    depth_mm: Optional[np.ndarray] = None,
) -> list[DoorRoi]:
    """Fallback ROIs when no complete door frame is detected."""
    img_h, img_w = bgr.shape[:2]
    rois: list[DoorRoi] = []

    if depth_mm is not None and depth_mm.shape[:2] == (img_h, img_w):
        rois.extend(_protrusion_hint_rois(depth_mm, img_h, img_w))

    rois.extend(_rgb_edge_hint_rois(bgr))

    # Center band is a weak prior for demo booths.
    rois.append(
        DoorRoi(
            int(img_w * 0.05),
            int(img_h * 0.2),
            int(img_w * 0.55),
            int(img_h * 0.55),
            "center_band",
            0.12,
            {"note": "weak prior"},
        ).clamp(img_h, img_w)
    )

    # Deduplicate overlapping ROIs, keep higher confidence.
    rois.sort(key=lambda r: r.confidence, reverse=True)
    picked: list[DoorRoi] = []
    for roi in rois:
        if all(_iou(roi, p) < 0.55 for p in picked):
            picked.append(roi)
        if len(picked) >= 6:
            break
    return picked


def _iou(a: DoorRoi, b: DoorRoi) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(float(union), 1.0)


def select_search_rois(
    bgr: np.ndarray,
    depth_mm: Optional[np.ndarray] = None,
    *,
    door_conf_min: float = 0.15,
) -> tuple[list[DoorRoi], str]:
    """
    Return ROIs to search for the handle.

    Priority:
    1. Door stile ROIs if a full door-like frame is found
    2. Otherwise RGB+Depth handle candidate regions
    """
    stile_rois = find_door_stile_rois(bgr)
    if stile_rois and stile_rois[0].confidence >= door_conf_min:
        return stile_rois, "door_frame"

    fallback = find_handle_search_rois(bgr, depth_mm)
    return fallback, "handle_search"
