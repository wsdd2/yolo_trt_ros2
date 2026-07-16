# -*- coding: utf-8
"""RGB-D door handle detection orchestrator."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .depth_protrusion import detect_depth_protrusion
from .detector import HandleDetector, detect_door_handle
from .door_roi import DoorRoi, select_search_rois
from .template_match import detect_by_template
from .types import HandleDetection


class RGBDHandleDetector:
    """
    Primary: door ROI + depth protrusion (material-agnostic).
    Fallback: legacy RGB geometry / template (RGB-only images).
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.2,
        door_conf_min: float = 0.15,
        use_template_fallback: bool = True,
        use_legacy_rgb_fallback: bool = True,
        extrapolate_grasp: bool = False,
    ) -> None:
        self.min_confidence = min_confidence
        self.door_conf_min = door_conf_min
        self.use_template_fallback = use_template_fallback
        self.use_legacy_rgb_fallback = use_legacy_rgb_fallback
        self.extrapolate_grasp = extrapolate_grasp
        self._legacy = HandleDetector(
            min_confidence=min_confidence,
            use_template_fallback=False,
        )

    def detect(
        self,
        bgr: np.ndarray,
        depth_mm: Optional[np.ndarray] = None,
    ) -> HandleDetection:
        if bgr is None or bgr.size == 0:
            return HandleDetection(success=False, message="empty image")

        rois, roi_mode = select_search_rois(bgr, depth_mm, door_conf_min=self.door_conf_min)

        best: Optional[HandleDetection] = None
        best_rank = -1.0

        if depth_mm is not None and depth_mm.size > 0:
            depth_det = detect_depth_protrusion(
                bgr,
                depth_mm,
                rois,
                extrapolate=self.extrapolate_grasp,
            )
            depth_det.debug["roi_mode"] = roi_mode
            depth_det.debug["search_rois"] = rois
            rank = depth_det.confidence if depth_det.success else -1.0
            if rank > best_rank:
                best_rank = rank
                best = depth_det

        if best is not None and best.success and best.confidence >= self.min_confidence:
            return best

        if self.use_template_fallback:
            for roi in rois[:3]:
                sy, sx = roi.as_slice()
                crop = bgr[sy, sx].copy()
                tpl_det = detect_by_template(crop)
                if tpl_det.success:
                    tpl_det.grasp_point = (
                        tpl_det.grasp_point[0] + roi.x,
                        tpl_det.grasp_point[1] + roi.y,
                    )
                    if tpl_det.handle_contour is not None:
                        tpl_det.handle_contour = tpl_det.handle_contour + np.array(
                            [[roi.x, roi.y]], dtype=np.int32
                        )
                    if tpl_det.backplate_contour is not None:
                        tpl_det.backplate_contour = tpl_det.backplate_contour + np.array(
                            [[roi.x, roi.y]], dtype=np.int32
                        )
                    if tpl_det.lever_contour is not None:
                        tpl_det.lever_contour = tpl_det.lever_contour + np.array(
                            [[roi.x, roi.y]], dtype=np.int32
                        )
                    tpl_det.message = "template_in_roi"
                    tpl_det.debug["roi"] = roi
                    tpl_det.debug["roi_mode"] = roi_mode
                    if tpl_det.confidence >= 0.45:
                        return tpl_det

        if self.use_legacy_rgb_fallback:
            legacy_det = self._legacy.detect(bgr)
            legacy_det.debug["roi_mode"] = roi_mode
            legacy_det.debug["search_rois"] = rois
            if legacy_det.success and (
                best is None or legacy_det.confidence >= best.confidence
            ):
                return legacy_det

        if best is not None and best.success:
            return best

        return HandleDetection(
            success=False,
            message="rgbd handle not found",
            debug={"roi_mode": roi_mode, "search_rois": rois},
        )


def detect_door_handle_rgbd(
    bgr: np.ndarray,
    depth_mm: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> HandleDetection:
    return RGBDHandleDetector(**kwargs).detect(bgr, depth_mm)
