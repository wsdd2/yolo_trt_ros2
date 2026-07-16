# -*- coding: utf-8 -*-
"""Shared geometry helpers for handle contour analysis."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def normalize_line_direction(vx: float, vy: float) -> tuple[float, float]:
    norm = float(np.hypot(vx, vy))
    if norm < 1e-6:
        return 1.0, 0.0
    return vx / norm, vy / norm


def contour_centroid(cnt: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(cnt)
    if abs(m["m00"]) < 1e-6:
        pts = cnt.reshape(-1, 2)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def mask_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def split_backplate_lever(
    handle_mask: np.ndarray,
    handle_contour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split handle blob into backplate (left/vertical) and lever (right) masks."""
    x, y, w, h = cv2.boundingRect(handle_contour)
    split_x = x + int(w * 0.38)
    backplate = np.zeros_like(handle_mask)
    lever = np.zeros_like(handle_mask)
    backplate[y : y + h, x:split_x] = handle_mask[y : y + h, x:split_x]
    lever[y : y + h, split_x : x + w] = handle_mask[y : y + h, split_x : x + w]

    bp_cnt = mask_contour(backplate)
    lever_cnt = mask_contour(lever)
    if bp_cnt is None:
        bp_cnt = handle_contour
    if lever_cnt is None:
        lever_cnt = handle_contour
    return backplate, lever, bp_cnt, lever_cnt


def grasp_from_lever_mask(
    lever_mask: np.ndarray,
    anchor_contour: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[Optional[tuple[int, int]], tuple[float, float, float, float]]:
    ys, xs = np.where(lever_mask > 0)
    if len(xs) < 6:
        return None, (1.0, 0.0, 0.0, 0.0)

    bx, by, bw, bh = cv2.boundingRect(anchor_contour)
    anchor_center = contour_centroid(anchor_contour)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    vx, vy = normalize_line_direction(vx, vy)

    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    if (cx - anchor_center[0]) * vx + (cy - anchor_center[1]) * vy < 0:
        vx, vy = -vx, -vy

    rel = pts - np.array(anchor_center, dtype=np.float64)
    proj = rel @ np.array([vx, vy], dtype=np.float64)
    valid = proj > float(max(bw, 8)) * 0.15
    if not np.any(valid):
        return None, (vx, vy, x0, y0)

    candidates = pts[valid]
    x_max = float(np.max(candidates[:, 0]))
    band = max(4.0, float(max(bw, 8)) * 0.2)
    right_cluster = candidates[candidates[:, 0] >= x_max - band]
    if len(right_cluster) < 2:
        cand_proj = (candidates - np.array(anchor_center, dtype=np.float64)) @ np.array([vx, vy], dtype=np.float64)
        top_idx = np.argsort(cand_proj)[-max(5, len(cand_proj) // 10) :]
        right_cluster = candidates[top_idx]

    grasp = (
        int(round(float(np.mean(right_cluster[:, 0])))),
        int(round(float(np.mean(right_cluster[:, 1])))),
    )

    margin = 6
    if not (margin <= grasp[0] < img_w - margin and margin <= grasp[1] < img_h - margin):
        return None, (vx, vy, x0, y0)

    dist = float(np.hypot(grasp[0] - anchor_center[0], grasp[1] - anchor_center[1]))
    if dist < float(max(bw, 8)) * 0.35 or dist > float(max(img_w, img_h)) * 0.45:
        return None, (vx, vy, x0, y0)

    return grasp, (vx, vy, x0, y0)


def extrapolate_grasp_horizontal(
    grasp: tuple[int, int],
    handle_mask: np.ndarray,
    anchor_contour: np.ndarray,
    img_w: int,
) -> tuple[int, int]:
    bx, by, bw, bh = cv2.boundingRect(anchor_contour)
    ys, xs = np.where(handle_mask > 0)
    if len(xs) < 6:
        return grasp

    x_vis = float(np.max(xs))
    visible_len = x_vis - (bx + bw * 0.4)
    expected_len = float(max(bw, 8)) * 2.1
    if visible_len >= expected_len * 0.88:
        tip_y = int(round(float(np.median(ys[xs >= x_vis - max(3, int(max(bw, 8) * 0.15))]))))
        return (int(round(x_vis)), tip_y)

    gx = int(round(min(img_w - 8, bx + bw * 0.4 + expected_len)))
    band = xs >= x_vis - max(3, int(max(bw, 8) * 0.15))
    gy = int(round(float(np.median(ys[band]))))
    if gx > grasp[0]:
        return (gx, gy)
    return grasp
