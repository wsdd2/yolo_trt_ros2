# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .types import HandleDetection


def _draw_search_rois(out: np.ndarray, det: HandleDetection) -> None:
    rois = det.debug.get("search_rois") or []
    active = det.debug.get("roi")
    for roi in rois:
        color = (180, 120, 255)
        if active is not None and roi.x == active.x and roi.y == active.y:
            color = (0, 255, 255)
        cv2.rectangle(out, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, 2)
        cv2.putText(
            out,
            roi.source,
            (roi.x + 4, max(roi.y + 16, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    roi_mode = det.debug.get("roi_mode")
    if roi_mode:
        cv2.putText(
            out,
            f"roi_mode={roi_mode}",
            (12, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 120, 255),
            1,
            cv2.LINE_AA,
        )


def draw_detection(
    bgr: np.ndarray,
    det: HandleDetection,
    *,
    show_axis: bool = True,
    show_geometry: bool = True,
    target_point: Optional[tuple[int, int]] = None,
    target_label: str = "grasp",
    extra_lines: Optional[list[str]] = None,
    show_target: bool = False,
    cached: bool = False,
    cache_info: str = "",
) -> np.ndarray:
    """Draw handle detection and the selected target point."""
    out = bgr.copy()
    if show_geometry:
        _draw_search_rois(out, det)

    if show_geometry and det.handle_contour is not None:
        cv2.drawContours(out, [det.handle_contour], -1, (0, 255, 0), 2)

    if show_geometry and det.backplate_contour is not None:
        cv2.drawContours(out, [det.backplate_contour], -1, (255, 180, 0), 2)

    if show_geometry and det.lever_contour is not None:
        cv2.drawContours(out, [det.lever_contour], -1, (0, 200, 255), 2)

    if show_geometry and show_axis and det.success and not cached:
        vx, vy, x0, y0 = det.lever_axis
        p0 = (int(x0), int(y0))
        scale = 120.0
        p1 = (int(x0 + vx * scale), int(y0 + vy * scale))
        cv2.arrowedLine(out, p0, p1, (255, 0, 255), 2, tipLength=0.2)

    point = target_point or (det.grasp_point if det.success else None)
    if point is not None and (det.success or show_target):
        gx, gy = point
        color = (0, 165, 255) if cached else (0, 0, 255)
        cv2.circle(out, (gx, gy), 10, color, 2)
        cv2.circle(out, (gx, gy), 4, color, -1)
        tag = "cache" if cached else target_label
        cv2.putText(
            out,
            f"{tag} ({gx}, {gy})",
            (gx + 12, gy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if cached:
        status = "CACHE"
        status_color = (0, 165, 255)
        status_conf = cache_info or f"inliers from last detections"
    else:
        status = "OK" if det.success else "FAIL"
        status_color = (0, 255, 0) if det.success else (0, 0, 255)
        status_conf = f"conf={det.confidence:.2f} {det.message}"
    cv2.putText(
        out,
        f"{status} {status_conf}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )
    if extra_lines:
        y = 58
        for line in extra_lines:
            cv2.putText(
                out,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24
    return out


def draw_debug_panel(
    bgr: np.ndarray,
    det: HandleDetection,
    panel_h: int = 180,
    *,
    show_geometry: bool = True,
    target_point: Optional[tuple[int, int]] = None,
    target_label: str = "grasp",
    extra_lines: Optional[list[str]] = None,
    show_target: bool = False,
    cached: bool = False,
    cache_info: str = "",
) -> np.ndarray:
    """Stack annotated image with debug mask (protrusion / metal)."""
    annotated = draw_detection(
        bgr,
        det,
        show_geometry=show_geometry,
        target_point=target_point,
        target_label=target_label,
        extra_lines=extra_lines,
        show_target=show_target,
        cached=cached,
        cache_info=cache_info,
    )
    mask = det.debug.get("protrusion_mask")
    if mask is None:
        mask = det.debug.get("metal_mask")
    if mask is None:
        return annotated

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    panel_w = int(w * panel_h / h)
    mask_small = cv2.resize(mask_bgr, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    ann_small = cv2.resize(annotated, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    row = np.hstack([ann_small, mask_small])
    return row
