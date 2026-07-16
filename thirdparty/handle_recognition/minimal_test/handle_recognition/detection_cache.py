# -*- coding: utf-8 -*-
"""Rolling detection cache for YOLOE live inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CacheObservation:
    u: int
    v: int
    confidence: float = 0.0
    p_cam: Optional[np.ndarray] = None


@dataclass
class CacheResult:
    success: bool
    center_px: tuple[int, int] = (0, 0)
    sample_count: int = 0
    inlier_count: int = 0
    message: str = ""


class DetectionCache:
    """Keep last N successful detections and return robust cluster center."""

    def __init__(
        self,
        *,
        max_frames: int = 30,
        min_samples: int = 3,
        max_dev_px: float = 80.0,
        max_dev_m: float = 0.12,
    ) -> None:
        self.max_frames = max(1, int(max_frames))
        self.min_samples = max(1, int(min_samples))
        self.max_dev_px = float(max_dev_px)
        self.max_dev_m = float(max_dev_m)
        self._history: deque[CacheObservation] = deque(maxlen=self.max_frames)

    def __len__(self) -> int:
        return len(self._history)

    def push(
        self,
        u: int,
        v: int,
        *,
        confidence: float = 0.0,
        p_cam: Optional[np.ndarray] = None,
    ) -> None:
        cam = None
        if p_cam is not None:
            cam = np.asarray(p_cam, dtype=np.float64).reshape(3)
        self._history.append(
            CacheObservation(u=int(u), v=int(v), confidence=float(confidence), p_cam=cam)
        )

    def clear(self) -> None:
        self._history.clear()

    def query(self) -> Optional[CacheResult]:
        if len(self._history) < self.min_samples:
            return None

        pts = np.array([[o.u, o.v] for o in self._history], dtype=np.float64)
        median_px = np.median(pts, axis=0)
        dist_px = np.linalg.norm(pts - median_px, axis=1)
        mad_px = float(np.median(dist_px))
        thresh_px = min(self.max_dev_px, max(12.0, mad_px * 2.5))
        inlier = dist_px <= thresh_px

        cam_list = [o.p_cam for o in self._history if o.p_cam is not None]
        if len(cam_list) >= self.min_samples:
            cam_pts = np.stack(cam_list, axis=0)
            median_cam = np.median(cam_pts, axis=0)
            dist_m = np.linalg.norm(cam_pts - median_cam, axis=1)
            mad_m = float(np.median(dist_m))
            thresh_m = min(self.max_dev_m, max(0.03, mad_m * 2.5))
            cam_inlier = dist_m <= thresh_m
            cam_indices = [i for i, o in enumerate(self._history) if o.p_cam is not None]
            for idx, ok in zip(cam_indices, cam_inlier):
                if not ok:
                    inlier[idx] = False

        inlier_pts = pts[inlier]
        if len(inlier_pts) < self.min_samples:
            return None

        center = np.mean(inlier_pts, axis=0)
        return CacheResult(
            success=True,
            center_px=(int(round(center[0])), int(round(center[1]))),
            sample_count=len(self._history),
            inlier_count=int(len(inlier_pts)),
            message="cache_cluster",
        )
