# -*- coding: utf-8 -*-
"""Depth sampling and pixel deprojection helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np


def median_valid_depth_mm(
    depth_mm: np.ndarray,
    u: int,
    v: int,
    *,
    radius: int = 3,
    min_mm: float = 100.0,
    max_mm: float = 5000.0,
) -> Optional[float]:
    """Return median depth (mm) in a square patch, ignoring invalid values."""
    h, w = depth_mm.shape[:2]
    x0 = max(0, u - radius)
    x1 = min(w, u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(h, v + radius + 1)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[(patch > min_mm) & (patch < max_mm)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def deproject_pixel_to_camera_m(
    u: int,
    v: int,
    depth_m: float,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Pinhole deprojection: pixel + depth -> 3D point in camera frame (meters)."""
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    x = (float(u) - cx) * depth_m / fx
    y = (float(v) - cy) * depth_m / fy
    return np.array([x, y, depth_m], dtype=np.float64)
