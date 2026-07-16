# -*- coding: utf-8 -*-
"""Template matching fallback for door handle localization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import cv2

from .types import HandleDetection


def _contour_centroid(cnt: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(cnt)
    if abs(m["m00"]) < 1e-6:
        pts = cnt.reshape(-1, 2)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])

_TEMPLATE_PATH = Path(__file__).resolve().parent / "assets" / "handle_template.png"
_GRASP_IN_TEMPLATE = (395, 114)  # right-end grasp point in template pixel coords


def _default_template_bgr() -> np.ndarray:
    """Build a synthetic lever-handle template if no asset is present."""
    tpl = np.full((150, 180, 3), 210, dtype=np.uint8)
    cv2.rectangle(tpl, (12, 28), (58, 122), (175, 178, 182), -1)
    cv2.rectangle(tpl, (52, 58), (165, 92), (165, 170, 175), -1)
    cv2.circle(tpl, _GRASP_IN_TEMPLATE, 5, (120, 125, 130), -1)
    return tpl


def load_handle_template() -> tuple[np.ndarray, tuple[int, int]]:
    if _TEMPLATE_PATH.is_file():
        tpl = cv2.imread(str(_TEMPLATE_PATH))
        if tpl is not None:
            return tpl, _GRASP_IN_TEMPLATE
    return _default_template_bgr(), _GRASP_IN_TEMPLATE


def build_template_from_image(bgr: np.ndarray, backplate: np.ndarray, handle_mask: np.ndarray) -> np.ndarray:
    """Crop a normalized template around a successful detection."""
    bx, by, bw, bh = cv2.boundingRect(backplate)
    x0 = max(0, bx - int(bw * 0.4))
    y0 = max(0, by - int(bh * 0.5))
    x1 = min(bgr.shape[1], bx + int(bw * 6.0))
    y1 = min(bgr.shape[0], by + int(bh * 1.6))
    crop = bgr[y0:y1, x0:x1].copy()
    mask = handle_mask[y0:y1, x0:x1]
    crop[mask == 0] = (crop[mask == 0] * 0.35 + 180 * 0.65).astype(np.uint8)
    return crop


def detect_by_template(
    bgr: np.ndarray,
    template_bgr: Optional[np.ndarray] = None,
    grasp_in_template: Optional[tuple[int, int]] = None,
) -> HandleDetection:
    """Multi-scale template matching fallback."""
    template, default_grasp = load_handle_template()
    if template_bgr is not None:
        template = template_bgr
    grasp_tpl = grasp_in_template or default_grasp

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    best_val = -1.0
    best_loc = (0, 0)
    best_scale = 1.0
    best_size = (tpl_gray.shape[1], tpl_gray.shape[0])

    for scale in (0.35, 0.45, 0.55, 0.65, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5, 1.8, 2.1):
        w = max(24, int(tpl_gray.shape[1] * scale))
        h = max(24, int(tpl_gray.shape[0] * scale))
        if w >= gray.shape[1] or h >= gray.shape[0]:
            continue
        resized = cv2.resize(tpl_gray, (w, h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = float(max_val)
            best_loc = max_loc
            best_scale = scale
            best_size = (w, h)

    if best_val < 0.38:
        return HandleDetection(success=False, message=f"template match weak ({best_val:.2f})")

    gx = int(best_loc[0] + grasp_tpl[0] * best_size[0] / tpl_gray.shape[1])
    gy = int(best_loc[1] + grasp_tpl[1] * best_size[1] / tpl_gray.shape[0])
    x, y = best_loc
    w, h = best_size
    handle_contour = np.array(
        [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]],
        dtype=np.int32,
    )
    backplate_w = max(8, int(w * 0.28))
    backplate_contour = np.array(
        [[[x, y + int(h * 0.15)]], [[x + backplate_w, y + int(h * 0.15)]],
         [[x + backplate_w, y + int(h * 0.85)]], [[x, y + int(h * 0.85)]]],
        dtype=np.int32,
    )
    lever_contour = np.array(
        [[[x + backplate_w, y + int(h * 0.35)]], [[x + w, y + int(h * 0.35)]],
         [[x + w, y + int(h * 0.65)]], [[x + backplate_w, y + int(h * 0.65)]]],
        dtype=np.int32,
    )
    bp_center = _contour_centroid(backplate_contour)
    return HandleDetection(
        success=True,
        grasp_point=(gx, gy),
        handle_contour=handle_contour,
        backplate_contour=backplate_contour,
        lever_contour=lever_contour,
        lever_axis=(1.0, 0.0, float(gx), float(gy)),
        backplate_center=bp_center,
        confidence=float(min(1.0, best_val)),
        message="template",
        debug={"match_score": best_val, "match_scale": best_scale, "match_loc": best_loc},
    )


def ensure_template_asset(reference_bgr: np.ndarray, backplate: np.ndarray, handle_mask: np.ndarray) -> None:
    """Persist a real template crop for future matching."""
    _TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tpl = build_template_from_image(reference_bgr, backplate, handle_mask)
    cv2.imwrite(str(_TEMPLATE_PATH), tpl)
