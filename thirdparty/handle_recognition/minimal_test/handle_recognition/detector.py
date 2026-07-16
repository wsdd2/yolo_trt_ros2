# -*- coding: utf-8 -*-
"""OpenCV-based door handle detection for grasp-point localization."""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from .template_match import detect_by_template
from .types import HandleDetection


def _normalize_line_direction(vx: float, vy: float) -> tuple[float, float]:
    norm = float(np.hypot(vx, vy))
    if norm < 1e-6:
        return 1.0, 0.0
    return vx / norm, vy / norm


def _contour_centroid(cnt: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(cnt)
    if abs(m["m00"]) < 1e-6:
        pts = cnt.reshape(-1, 2)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def _metal_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    mask_hsv = cv2.inRange(hsv, (0, 0, 75), (180, 85, 245))
    mask_lab = cv2.inRange(lab, (0, 118, 118), (255, 142, 142))
    mask_gray = cv2.inRange(gray, 95, 215)
    combined = cv2.bitwise_and(mask_hsv, mask_lab)
    combined = cv2.bitwise_and(combined, mask_gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined


def _refine_metal_mask(metal: np.ndarray, erosion_iters: int) -> np.ndarray:
    if erosion_iters <= 0:
        return metal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.erode(metal, kernel, iterations=erosion_iters)
    return cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)


def _mean_saturation(bgr: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 255.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1][mask > 0]))


def _backplate_score(cnt: np.ndarray, bgr: np.ndarray, img_h: int, img_w: int) -> float:
    area = float(cv2.contourArea(cnt))
    if area < 300.0 or area > img_h * img_w * 0.05:
        return 0.0

    x, y, w, h = cv2.boundingRect(cnt)
    if h > img_h * 0.22 or w > img_w * 0.12 or w > 70:
        return 0.0
    if h < 16 or w < 8:
        return 0.0

    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    if min(rw, rh) < 1.0:
        return 0.0
    long_side = max(rw, rh)
    short_side = min(rw, rh)
    aspect = long_side / short_side
    if aspect < 1.2 or aspect > 6.0:
        return 0.0

    box_area = long_side * short_side
    rectangularity = area / box_area if box_area > 0 else 0.0
    if rectangularity < 0.45:
        return 0.0

    angle = abs(rect[2])
    if long_side == rw:
        angle = abs(90.0 - angle)
    vertical_bonus = 1.0 - min(angle, 90.0 - min(angle, 90.0)) / 45.0

    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    sat = _mean_saturation(bgr, mask)
    if sat > 100.0:
        return 0.0
    metal_bonus = max(0.0, 1.0 - sat / 100.0)

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
    poly_bonus = 1.12 if len(approx) == 4 else 1.0

    return area * rectangularity * (0.45 + 0.55 * vertical_bonus) * (0.4 + 0.6 * metal_bonus) * poly_bonus


def _find_backplate(
    contours: list[np.ndarray],
    bgr: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[Optional[np.ndarray], float]:
    best: Optional[np.ndarray] = None
    best_score = 0.0
    for cnt in contours:
        score = _backplate_score(cnt, bgr, img_h, img_w)
        if score > best_score:
            best_score = score
            best = cnt
    return best, best_score


def _connected_handle_mask(
    metal: np.ndarray,
    backplate: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    bx, by, bw, bh = cv2.boundingRect(backplate)
    pad_x = int(max(bw * 3.5, 90))
    pad_y = int(max(bh * 1.3, 45))
    x0 = max(0, bx - int(bw * 0.2))
    y0 = max(0, by - pad_y)
    x1 = min(img_w, bx + bw + pad_x)
    y1 = min(img_h, by + bh + pad_y)

    roi = metal[y0:y1, x0:x1].copy()
    roi_h, roi_w = roi.shape[:2]
    if roi.size == 0:
        return np.zeros_like(metal)

    num, labels, _, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if num <= 1:
        return np.zeros_like(metal)

    bp_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    bp_shift = backplate - np.array([[x0, y0]], dtype=np.int32)
    cv2.drawContours(bp_mask, [bp_shift], -1, 255, thickness=cv2.FILLED)
    bp_pixels = labels[bp_mask > 0]
    if bp_pixels.size == 0:
        return np.zeros_like(metal)

    keep_label = int(np.bincount(bp_pixels).argmax())
    handle_roi = np.zeros_like(roi)
    handle_roi[labels == keep_label] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    handle_roi = cv2.morphologyEx(handle_roi, cv2.MORPH_CLOSE, kernel, iterations=2)

    out = np.zeros_like(metal)
    out[y0:y1, x0:x1] = handle_roi
    # Drop latch / strike plate to the left of the backplate.
    out[:, : max(0, bx - int(bw * 0.15))] = 0
    return out


def _lever_mask_from_handle(handle_mask: np.ndarray, backplate: np.ndarray) -> np.ndarray:
    bx, by, bw, bh = cv2.boundingRect(backplate)
    lever = np.zeros_like(handle_mask)
    x0 = bx + int(bw * 0.45)
    y0 = by - int(bh * 0.25)
    y1 = by + bh + int(bh * 0.55)
    lever[y0:y1, x0:] = handle_mask[y0:y1, x0:]
    return lever


def _mask_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _grasp_from_lever_mask(
    lever_mask: np.ndarray,
    backplate: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[Optional[tuple[int, int]], tuple[float, float, float, float]]:
    ys, xs = np.where(lever_mask > 0)
    if len(xs) < 8:
        return None, (1.0, 0.0, 0.0, 0.0)

    bx, by, bw, bh = cv2.boundingRect(backplate)
    backplate_center = _contour_centroid(backplate)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    vx, vy = _normalize_line_direction(vx, vy)

    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    if (cx - backplate_center[0]) * vx + (cy - backplate_center[1]) * vy < 0:
        vx, vy = -vx, -vy

    rel = pts - np.array(backplate_center, dtype=np.float64)
    proj = rel @ np.array([vx, vy], dtype=np.float64)
    valid = proj > float(bw) * 0.2
    if not np.any(valid):
        return None, (vx, vy, x0, y0)

    candidates = pts[valid]
    x_max = float(np.max(candidates[:, 0]))
    band = max(4.0, float(bw) * 0.18)
    right_cluster = candidates[candidates[:, 0] >= x_max - band]
    if len(right_cluster) < 2:
        cand_proj = (candidates - np.array(backplate_center, dtype=np.float64)) @ np.array([vx, vy], dtype=np.float64)
        top_idx = np.argsort(cand_proj)[-max(5, len(cand_proj) // 10) :]
        right_cluster = candidates[top_idx]
    grasp = (
        int(round(float(np.mean(right_cluster[:, 0])))),
        int(round(float(np.mean(right_cluster[:, 1])))),
    )

    margin = 6
    if not (margin <= grasp[0] < img_w - margin and margin <= grasp[1] < img_h - margin):
        return None, (vx, vy, x0, y0)

    dist = float(np.hypot(grasp[0] - backplate_center[0], grasp[1] - backplate_center[1]))
    if dist < float(bw) * 0.4 or dist > float(max(img_w, img_h)) * 0.42:
        return None, (vx, vy, x0, y0)

    return grasp, (vx, vy, x0, y0)


def _refine_grasp_with_handle_contour(
    grasp: tuple[int, int],
    lever_axis: tuple[float, float, float, float],
    handle_contour: np.ndarray,
    backplate: np.ndarray,
) -> tuple[int, int]:
    """Push grasp toward the visible right end of the full handle contour."""
    bx, by, bw, bh = cv2.boundingRect(backplate)
    backplate_center = _contour_centroid(backplate)
    vx, vy = lever_axis[0], lever_axis[1]
    pts = handle_contour.reshape(-1, 2).astype(np.float64)
    rel = pts - np.array(backplate_center, dtype=np.float64)
    proj = rel @ np.array([vx, vy], dtype=np.float64)
    right_pts = pts[(proj > float(bw) * 0.45) & (pts[:, 0] > backplate_center[0])]
    if len(right_pts) < 4:
        return grasp

    x_max = float(np.max(right_pts[:, 0]))
    band = max(5.0, float(bw) * 0.22)
    tip_cluster = right_pts[right_pts[:, 0] >= x_max - band]
    if len(tip_cluster) < 2:
        return grasp
    refined = (
        int(round(float(np.mean(tip_cluster[:, 0])))),
        int(round(float(np.mean(tip_cluster[:, 1])))),
    )
    if refined[0] <= grasp[0]:
        return grasp
    return refined


def _extrapolate_grasp_if_short(
    grasp: tuple[int, int],
    lever_axis: tuple[float, float, float, float],
    backplate: np.ndarray,
    lever_mask: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[int, int]:
    """Extend grasp along lever axis when visible contour ends before the tip."""
    bx, by, bw, bh = cv2.boundingRect(backplate)
    if abs(lever_axis[0]) < abs(lever_axis[1]):
        return grasp

    ys, _ = np.where(lever_mask > 0)
    if len(ys) == 0:
        return grasp
    lever_y = float(np.median(ys))

    origin = np.array([bx + bw, lever_y], dtype=np.float64)
    vx, vy = lever_axis[0], lever_axis[1]
    dist = float(np.hypot(grasp[0] - origin[0], grasp[1] - origin[1]))
    target = float(bw) * 2.35
    if dist >= target * 0.85:
        return grasp

    ext = origin + np.array([vx, vy], dtype=np.float64) * target
    gx = int(round(ext[0]))
    gy = int(round(ext[1]))
    if abs(gy - grasp[1]) > max(12.0, float(bh) * 0.45):
        gy = int(round(grasp[1]))

    margin = 6
    if margin <= gx < img_w - margin and margin <= gy < img_h - margin:
        return (gx, gy)
    return grasp


def _extrapolate_horizontal_fallback(
    grasp: tuple[int, int],
    handle_mask: np.ndarray,
    backplate: np.ndarray,
    img_w: int,
) -> tuple[int, int]:
    """When lever over glass is missing, extend grasp to the right from visible metal."""
    bx, by, bw, bh = cv2.boundingRect(backplate)
    ys, xs = np.where(handle_mask > 0)
    if len(xs) < 8:
        return grasp

    x_vis = float(np.max(xs))
    visible_len = x_vis - (bx + bw * 0.45)
    expected_len = float(bw) * 2.15
    if visible_len >= expected_len * 0.9:
        tip_y = int(round(float(np.median(ys[xs >= x_vis - max(3, int(bw * 0.15))]))))
        return (int(round(x_vis)), tip_y)

    gx = int(round(min(img_w - 8, bx + bw * 0.45 + expected_len)))
    band = xs >= x_vis - max(3, int(bw * 0.15))
    gy = int(round(float(np.median(ys[band]))))
    if gx > grasp[0]:
        return (gx, gy)
    return grasp


def _geometry_rank(det: HandleDetection) -> float:
    """Prefer compact backplates and stable lever spans."""
    if not det.success or det.backplate_contour is None:
        return -1.0
    bx, by, bw, bh = cv2.boundingRect(det.backplate_contour)
    bp_area = float(bw * bh)
    compact = 1.0 / (1.0 + bp_area / 2500.0)
    gx, gy = det.grasp_point
    cx, cy = det.backplate_center
    if gx <= cx:
        return -1.0
    return det.confidence * compact


def _detect_geometry_once(
    bgr: np.ndarray,
    metal: np.ndarray,
    *,
    min_backplate_area: float,
) -> HandleDetection:
    img_h, img_w = bgr.shape[:2]
    contours, _ = cv2.findContours(metal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_backplate_area]
    if not contours:
        return HandleDetection(success=False, message="no metal contour found", debug={"metal_mask": metal})

    backplate, bp_score = _find_backplate(contours, bgr, img_h, img_w)
    if backplate is None or bp_score <= 0.0:
        return HandleDetection(success=False, message="backplate not found", debug={"metal_mask": metal})

    handle_mask = _connected_handle_mask(metal, backplate, img_h, img_w)
    handle_contour = _mask_contour(handle_mask)
    if handle_contour is None or cv2.contourArea(handle_contour) < min_backplate_area:
        return HandleDetection(success=False, message="handle mask empty", debug={"metal_mask": metal})

    lever_mask = _lever_mask_from_handle(handle_mask, backplate)
    lever_contour = _mask_contour(lever_mask)
    if lever_contour is None:
        return HandleDetection(success=False, message="lever mask empty", debug={"metal_mask": metal})

    backplate_center = _contour_centroid(backplate)
    grasp_point, lever_axis = _grasp_from_lever_mask(lever_mask, backplate, img_h, img_w)
    if grasp_point is None:
        return HandleDetection(
            success=False,
            message="grasp point invalid",
            handle_contour=handle_contour,
            backplate_contour=backplate,
            lever_contour=lever_contour,
            backplate_center=backplate_center,
            debug={"metal_mask": metal, "handle_mask": handle_mask, "lever_mask": lever_mask},
        )

    grasp_point = _refine_grasp_with_handle_contour(
        grasp_point, lever_axis, handle_contour, backplate
    )
    grasp_point = _extrapolate_grasp_if_short(
        grasp_point, lever_axis, backplate, lever_mask, img_h, img_w
    )
    grasp_point = _extrapolate_horizontal_fallback(
        grasp_point, handle_mask, backplate, img_w
    )

    bx, by, bw, bh = cv2.boundingRect(backplate)
    lever_span = float(cv2.boundingRect(lever_contour)[2])
    span_ratio = lever_span / max(float(bw), 1.0)
    confidence = float(min(1.0, 0.42 * min(bp_score / 3500.0, 1.0) + 0.58 * min(span_ratio / 1.8, 1.0)))

    return HandleDetection(
        success=True,
        grasp_point=grasp_point,
        handle_contour=handle_contour,
        backplate_contour=backplate,
        lever_contour=lever_contour,
        lever_axis=lever_axis,
        backplate_center=backplate_center,
        confidence=confidence,
        message="geometry",
        debug={
            "metal_mask": metal,
            "handle_mask": handle_mask,
            "lever_mask": lever_mask,
            "backplate_score": bp_score,
        },
    )


class HandleDetector:
    """Detect lever handle contour and right-end grasp point in a BGR image."""

    def __init__(
        self,
        *,
        min_backplate_area: float = 300.0,
        min_confidence: float = 0.2,
        use_template_fallback: bool = True,
    ) -> None:
        self.min_backplate_area = min_backplate_area
        self.min_confidence = min_confidence
        self.use_template_fallback = use_template_fallback

    def detect(self, bgr: np.ndarray) -> HandleDetection:
        if bgr is None or bgr.size == 0:
            return HandleDetection(success=False, message="empty image")

        best_geom: Optional[HandleDetection] = None
        best_rank = -1.0
        base_metal = _metal_mask(bgr)
        for erosion_iters in (0, 1, 2):
            metal = _refine_metal_mask(base_metal, erosion_iters)
            det = _detect_geometry_once(bgr, metal, min_backplate_area=self.min_backplate_area)
            det.debug["erosion_iters"] = erosion_iters
            rank = _geometry_rank(det)
            if rank > best_rank:
                best_rank = rank
                best_geom = det

        if best_geom is not None and best_geom.success and best_geom.confidence >= self.min_confidence:
            return best_geom

        if self.use_template_fallback:
            tpl_det = detect_by_template(bgr)
            if tpl_det.success and tpl_det.confidence >= 0.45:
                return tpl_det

        if best_geom is not None and best_geom.success:
            return best_geom

        return HandleDetection(success=False, message="handle not found", debug={"metal_mask": base_metal})


def detect_door_handle(bgr: np.ndarray, **kwargs: Any) -> HandleDetection:
    """Convenience wrapper."""
    return HandleDetector(**kwargs).detect(bgr)
