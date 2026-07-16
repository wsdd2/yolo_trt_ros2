# -*- coding: utf-8 -*-
"""Electrical-cabinet control semantics + OpenCV ROI / center refinement.

Shared by:
  - handle_recognition/minimal_test (direct H2 script)
  - Foxy_ROS yolo_trt_ros2 (ROS detector / projector)

Keep free of ROS and h2_pipeline imports so either path can import it alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

# YOLOE set_classes prompts (English works best with MobileCLIP).
DEFAULT_CABINET_PROMPTS: tuple[str, ...] = (
    "red indicator light",
    "green indicator light",
    "yellow indicator light",
    "black indicator light",
    "red push button",
    "green push button",
    "yellow push button",
    "black push button",
    "square analog ammeter",
    "square analog voltmeter",
    "digital panel meter",
    "black digital display",
    "multi-color indicator panel",
    "black rotary selector switch",
    "rotary knob with pointer",
    "black rocker switch",
    "yellow toggle switch",
    "red toggle switch",
    "white toggle switch",
    "round pressure gauge",
    "circular analog gauge",
    "white text label",
)

# Map raw YOLOE / open-vocab labels -> stable publish class_name.
LABEL_TO_CLASS: dict[str, str] = {
    "red indicator light": "indicator_light",
    "green indicator light": "indicator_light",
    "yellow indicator light": "indicator_light",
    "black indicator light": "indicator_light",
    "indicator light": "indicator_light",
    "push button": "indicator_light",
    "red push button": "indicator_light",
    "green push button": "indicator_light",
    "yellow push button": "indicator_light",
    "black push button": "indicator_light",
    "square analog ammeter": "analog_meter",
    "square analog voltmeter": "analog_meter",
    "analog ammeter": "analog_meter",
    "analog voltmeter": "analog_meter",
    "analog meter": "analog_meter",
    "digital panel meter": "digital_meter",
    "black digital display": "digital_meter",
    "digital meter": "digital_meter",
    "multi-color indicator panel": "indicator_panel",
    "black rotary selector switch": "rotary_switch",
    "rotary knob with pointer": "rotary_switch",
    "rotary selector knob": "rotary_switch",
    "selector switch": "rotary_switch",
    "rotary switch": "rotary_switch",
    "black rocker switch": "rocker_switch",
    "rocker switch": "rocker_switch",
    "rectangular rocker breaker": "rocker_switch",
    "yellow toggle switch": "toggle_switch",
    "red toggle switch": "toggle_switch",
    "white toggle switch": "toggle_switch",
    "toggle switch": "toggle_switch",
    "lever switch": "toggle_switch",
    "round pressure gauge": "pressure_gauge",
    "circular analog gauge": "pressure_gauge",
    "analog pressure gauge": "pressure_gauge",
    "pressure gauge": "pressure_gauge",
    "white text label": "label",
    "text label": "label",
    "nameplate label": "label",
}

# Classes that are typically press / flip / turn interaction targets.
PRESSABLE_CLASSES = frozenset(
    {
        "indicator_light",
        "rotary_switch",
        "rocker_switch",
        "toggle_switch",
    }
)

# Default world-frame contact offset after depth projection (meters).
# Same convention as blue_point_target_world_offset: tip contact slightly into panel.
DEFAULT_CLASS_WORLD_OFFSETS_M: dict[str, tuple[float, float, float]] = {
    "indicator_light": (0.0, 0.0, -0.003),
    "rotary_switch": (0.0, 0.0, -0.002),
    "rocker_switch": (0.0, 0.0, -0.003),
    "toggle_switch": (0.0, 0.0, -0.002),
    "analog_meter": (0.0, 0.0, 0.0),
    "digital_meter": (0.0, 0.0, 0.0),
    "indicator_panel": (0.0, 0.0, 0.0),
    "pressure_gauge": (0.0, 0.0, 0.0),
    "label": (0.0, 0.0, 0.0),
}

CIRCLE_REFINE_CLASSES = frozenset({"indicator_light", "rotary_switch", "pressure_gauge"})
RECT_REFINE_CLASSES = frozenset({"analog_meter", "digital_meter", "rocker_switch"})
ROW_SNAP_CLASSES = frozenset({"indicator_light", "toggle_switch"})


@dataclass
class PanelRoi:
    xyxy: tuple[int, int, int, int]
    source: str = "opencv"

    def to_list(self) -> list[int]:
        return [int(v) for v in self.xyxy]


def load_prompts_file(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    prompts: list[str] = []
    for raw in text.replace(",", "\n").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def infer_color(label: str) -> Optional[str]:
    name = label.lower()
    for color in ("red", "green", "yellow", "black", "white"):
        if color in name:
            return color
    return None


def normalize_cabinet_label(label: str) -> tuple[str, Optional[str]]:
    """Return (canonical_class, color_hint)."""
    text = str(label or "").lower().strip().replace(".", "")
    color = infer_color(text)
    best = ""
    for key, cls_name in LABEL_TO_CLASS.items():
        if key in text:
            best = cls_name
            break
    if not best:
        if "button" in text or "light" in text or "lamp" in text:
            best = "indicator_light"
        elif "knob" in text or "selector" in text or "rotary" in text:
            best = "rotary_switch"
        elif "toggle" in text or "lever" in text:
            best = "toggle_switch"
        elif "rocker" in text:
            best = "rocker_switch"
        elif "gauge" in text:
            best = "pressure_gauge"
        elif "ammeter" in text or "voltmeter" in text or "meter" in text:
            best = "analog_meter" if "digital" not in text else "digital_meter"
        elif "label" in text or "text" in text:
            best = "label"
        else:
            best = text.replace(" ", "_") or "unknown"
    return best, color


def class_world_offset_m(
    class_name: str,
    *,
    overrides: Optional[dict[str, Sequence[float]]] = None,
    press_offset: Optional[Sequence[float]] = None,
) -> np.ndarray:
    name = str(class_name or "").strip().lower()
    if overrides and name in overrides:
        return np.asarray(overrides[name], dtype=np.float64).reshape(3)
    if press_offset is not None and name in PRESSABLE_CLASSES:
        return np.asarray(press_offset, dtype=np.float64).reshape(3)
    if name in DEFAULT_CLASS_WORLD_OFFSETS_M:
        return np.asarray(DEFAULT_CLASS_WORLD_OFFSETS_M[name], dtype=np.float64).reshape(3)
    return np.zeros(3, dtype=np.float64)


def estimate_panel_roi(
    bgr: np.ndarray,
    *,
    depth_mm: Optional[np.ndarray] = None,
    min_area_ratio: float = 0.18,
) -> Optional[PanelRoi]:
    """Estimate the dominant gray metal cabinet face as an axis-aligned ROI."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Light industrial gray panel: low saturation, mid value.
    gray = cv2.inRange(hsv, (0, 0, 70), (180, 55, 210))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    if depth_mm is not None and depth_mm.shape[:2] == (h, w):
        valid = (depth_mm > 200) & (depth_mm < 2500)
        if int(np.count_nonzero(valid)) > 500:
            vals = depth_mm[valid].astype(np.float32)
            med = float(np.median(vals))
            band = np.abs(depth_mm.astype(np.float32) - med) < 80.0
            gray = cv2.bitwise_and(gray, (band & valid).astype(np.uint8) * 255)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    if area < float(w * h) * float(min_area_ratio):
        return None
    x, y, bw, bh = cv2.boundingRect(best)
    pad_x = int(round(bw * 0.02))
    pad_y = int(round(bh * 0.02))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w - 1, x + bw + pad_x)
    y2 = min(h - 1, y + bh + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return PanelRoi(xyxy=(x1, y1, x2, y2), source="opencv_gray_panel")


def clip_bbox_to_image(
    xyxy: Sequence[int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def bbox_center_inside_roi(xyxy: Sequence[int], roi: Sequence[int]) -> bool:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    rx1, ry1, rx2, ry2 = [int(v) for v in roi]
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _fit_circle_in_roi(bgr: np.ndarray, xyxy: Sequence[int]) -> Optional[tuple[int, int, int]]:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = clip_bbox_to_image(xyxy, w, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 140)
    min_r = max(3, int(round(min(crop.shape[0], crop.shape[1]) * 0.18)))
    max_r = max(min_r + 1, int(round(min(crop.shape[0], crop.shape[1]) * 0.55)))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, min_r),
        param1=120,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        # Fallback: largest circular contour by circularity.
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 20:
                continue
            peri = float(cv2.arcLength(cnt, True))
            if peri <= 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (peri * peri)
            if circularity < 0.55:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            score = circularity * area
            if score > best_score:
                best_score = score
                best = (int(round(cx)), int(round(cy)), int(round(radius)))
        if best is None:
            return None
        cx, cy, radius = best
    else:
        arr = np.round(circles[0]).astype(int)
        # Prefer circle closest to crop center.
        cx0 = (crop.shape[1] - 1) * 0.5
        cy0 = (crop.shape[0] - 1) * 0.5
        best_i = int(np.argmin((arr[:, 0] - cx0) ** 2 + (arr[:, 1] - cy0) ** 2))
        cx, cy, radius = int(arr[best_i, 0]), int(arr[best_i, 1]), int(arr[best_i, 2])
    return x1 + cx, y1 + cy, max(1, radius)


def _fit_rect_center(bgr: np.ndarray, xyxy: Sequence[int]) -> Optional[tuple[int, int, tuple[int, int, int, int]]]:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = clip_bbox_to_image(xyxy, w, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 25:
        return None
    rx, ry, rw, rh = cv2.boundingRect(best)
    cx = x1 + rx + rw // 2
    cy = y1 + ry + rh // 2
    refined = (x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh)
    return cx, cy, refined


def refine_detection_geometry(
    bgr: np.ndarray,
    *,
    class_name: str,
    bbox_xyxy: Sequence[int],
    center_px: Sequence[int],
) -> dict[str, Any]:
    """OpenCV secondary compensation for YOLOE boxes."""
    meta: dict[str, Any] = {
        "refine_source": "none",
        "center_px_raw": [int(center_px[0]), int(center_px[1])],
    }
    cls = str(class_name).lower()
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cx, cy = int(center_px[0]), int(center_px[1])

    if cls in CIRCLE_REFINE_CLASSES:
        fitted = _fit_circle_in_roi(bgr, bbox_xyxy)
        if fitted is not None:
            fcx, fcy, radius = fitted
            cx, cy = fcx, fcy
            # Tighten bbox around fitted circle.
            x1 = max(0, fcx - radius)
            y1 = max(0, fcy - radius)
            x2 = fcx + radius
            y2 = fcy + radius
            meta["refine_source"] = "opencv_circle"
            meta["circle_radius_px"] = int(radius)
    elif cls in RECT_REFINE_CLASSES:
        fitted = _fit_rect_center(bgr, bbox_xyxy)
        if fitted is not None:
            cx, cy, refined = fitted
            x1, y1, x2, y2 = refined
            meta["refine_source"] = "opencv_rect"

    meta["center_px"] = [int(cx), int(cy)]
    meta["bbox_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]
    return meta


def snap_row_centers(
    items: list[dict[str, Any]],
    *,
    y_tol_px: float = 18.0,
) -> list[dict[str, Any]]:
    """Snap Y of same-class detections that form a horizontal row."""
    by_class: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        cls = str(item.get("class_name", ""))
        if cls not in ROW_SNAP_CLASSES:
            continue
        by_class.setdefault(cls, []).append(i)

    for indices in by_class.values():
        if len(indices) < 3:
            continue
        ys = np.asarray([float(items[i]["center_px"][1]) for i in indices], dtype=np.float64)
        # Greedy cluster by Y.
        order = list(np.argsort(ys))
        clusters: list[list[int]] = []
        for idx in order:
            y = float(ys[idx])
            placed = False
            for cluster in clusters:
                cy = float(np.mean([ys[j] for j in cluster]))
                if abs(y - cy) <= y_tol_px:
                    cluster.append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([idx])
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            mean_y = int(round(float(np.mean([ys[j] for j in cluster]))))
            for local_i in cluster:
                real_i = indices[local_i]
                cx = int(items[real_i]["center_px"][0])
                items[real_i]["center_px"] = [cx, mean_y]
                items[real_i]["row_snap_y"] = mean_y
                items[real_i]["refine_source"] = (
                    str(items[real_i].get("refine_source", "none")) + "+row_snap"
                )
    return items


def assign_panel_rows(
    items: list[dict[str, Any]],
    *,
    roi: Optional[Sequence[int]] = None,
    n_rows: int = 5,
) -> list[dict[str, Any]]:
    """Assign coarse panel_row index (1..n_rows) by vertical bands inside ROI."""
    if not items:
        return items
    if roi is None:
        ys = [int(it["center_px"][1]) for it in items]
        y1, y2 = min(ys), max(ys)
    else:
        y1, y2 = int(roi[1]), int(roi[3])
    span = max(1, y2 - y1)
    for it in items:
        cy = int(it["center_px"][1])
        rel = (cy - y1) / float(span)
        row = int(np.clip(np.floor(rel * n_rows) + 1, 1, n_rows))
        it["panel_row"] = int(row)
    return items


def filter_allowed_classes(
    class_name: str,
    *,
    allow_labels: bool = False,
) -> bool:
    cls = str(class_name).lower()
    if cls == "label":
        return bool(allow_labels)
    if cls == "unknown":
        return False
    return cls in DEFAULT_CLASS_WORLD_OFFSETS_M


def prepare_cabinet_detections(
    raw_detections: Iterable[Any],
    bgr: np.ndarray,
    *,
    depth_mm: Optional[np.ndarray] = None,
    panel_roi: Optional[Sequence[int]] = None,
    auto_panel_roi: bool = True,
    allow_labels: bool = False,
    max_objects: int = 64,
    conf_min: float = 0.12,
) -> tuple[list[dict[str, Any]], Optional[PanelRoi]]:
    """Normalize YOLOE boxes, ROI-filter, refine centers, row-snap."""
    h, w = bgr.shape[:2]
    roi_obj: Optional[PanelRoi] = None
    if panel_roi is not None:
        roi_obj = PanelRoi(xyxy=clip_bbox_to_image(panel_roi, w, h), source="manual")
    elif auto_panel_roi:
        roi_obj = estimate_panel_roi(bgr, depth_mm=depth_mm)

    prepared: list[dict[str, Any]] = []
    for det in raw_detections:
        if hasattr(det, "class_name"):
            raw_name = str(det.class_name)
            conf = float(det.confidence)
            bbox = tuple(int(v) for v in det.bbox_xyxy)
            center = tuple(int(v) for v in det.center_px)
            source = str(getattr(det, "source", "yoloe"))
        else:
            raw_name = str(det.get("class_name", ""))
            conf = float(det.get("confidence", 0.0))
            bbox = tuple(int(v) for v in det.get("bbox_xyxy", (0, 0, 0, 0)))
            center = tuple(int(v) for v in det.get("center_px", (0, 0)))
            source = str(det.get("source", "yoloe"))

        if conf < conf_min:
            continue
        class_name, color = normalize_cabinet_label(raw_name)
        if not filter_allowed_classes(class_name, allow_labels=allow_labels):
            continue
        if roi_obj is not None and not bbox_center_inside_roi(bbox, roi_obj.xyxy):
            continue

        geom = refine_detection_geometry(
            bgr,
            class_name=class_name,
            bbox_xyxy=bbox,
            center_px=center,
        )
        prepared.append(
            {
                "raw_class_name": raw_name,
                "class_name": class_name,
                "color": color,
                "confidence": conf,
                "bbox_xyxy": geom["bbox_xyxy"],
                "center_px": geom["center_px"],
                "center_px_raw": geom["center_px_raw"],
                "source": source,
                "refine_source": geom.get("refine_source", "none"),
                "circle_radius_px": geom.get("circle_radius_px"),
            }
        )

    prepared.sort(key=lambda d: float(d["confidence"]), reverse=True)
    prepared = prepared[: max(1, int(max_objects))]
    prepared = snap_row_centers(prepared)
    prepared = assign_panel_rows(prepared, roi=None if roi_obj is None else roi_obj.xyxy)
    return prepared, roi_obj
