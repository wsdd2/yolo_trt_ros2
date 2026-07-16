# -*- coding: utf-8 -*-
"""Open-vocabulary door handle detection via Ultralytics YOLOE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLOE
except ImportError as exc:  # pragma: no cover
    YOLOE = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

DEFAULT_PROMPTS = (
    "lever door handle",
    "horizontal door handle",
    "door lever handle",
    "pull door handle",
)

HANDLE_KEYWORDS = (
    "lever",
    "pull",
    "handle",
)

LOCK_KEYWORDS = (
    "lock",
    "keyhole",
    "key",
    "deadbolt",
    "cylinder",
    "knob",
)


def _bbox_aspect(x1: int, y1: int, x2: int, y2: int) -> float:
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return float(w) / float(h)


def _lever_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """Use bar-center for horizontal lever, not lock plate below."""
    w = x2 - x1
    h = y2 - y1
    cx = int(round(x1 + w * 0.55))
    cy = int(round(y1 + h * 0.5))
    return cx, cy


def _score_lever_candidate(
    cls_name: str,
    conf: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_h: int,
    img_w: int,
) -> float:
    w = x2 - x1
    h = y2 - y1
    if w < 16 or h < 8:
        return 0.0

    name = cls_name.lower()
    if any(k in name for k in LOCK_KEYWORDS):
        return 0.0

    aspect = _bbox_aspect(x1, y1, x2, y2)
    score = conf

    # Lever is a long horizontal bar; lock/keyhole is compact and closer to square.
    if aspect < 1.15:
        score *= 0.15
    elif aspect < 1.35:
        score *= 0.45
    else:
        score *= 1.0 + min(0.8, (aspect - 1.35) * 0.25)

    if w < h * 1.1:
        score *= 0.2

    if "lever" in name or "pull" in name:
        score *= 1.4
    elif "handle" in name:
        score *= 1.15
    elif "knob" in name:
        score *= 0.25

    # When handle and lock are stacked vertically, prefer the upper long bar.
    cy = (y1 + y2) * 0.5
    if aspect >= 1.35:
        score *= 1.0 + 0.25 * (1.0 - min(1.0, cy / max(float(img_h), 1.0)))

    return score


@dataclass
class YOLOEHandleDetection:
    success: bool
    bbox_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)
    center_px: tuple[int, int] = (0, 0)
    confidence: float = 0.0
    class_name: str = ""
    message: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


class YOLOEHandleDetector:
    """Text-prompted or prompt-free YOLOE door handle detector."""

    def __init__(
        self,
        *,
        model_path: str = "yoloe-11s-seg.pt",
        prompts: tuple[str, ...] = DEFAULT_PROMPTS,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        prompt_free: bool = False,
    ) -> None:
        if YOLOE is None:
            raise RuntimeError(
                "未安装 ultralytics，请运行: pip install -r requirements-yoloe.txt"
            ) from _IMPORT_ERROR
        self.model_path = model_path
        self.prompts = tuple(p.strip() for p in prompts if p.strip())
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.prompt_free = prompt_free
        self.model = YOLOE(model_path)
        if not prompt_free and self.prompts:
            self.model.set_classes(list(self.prompts))

    def detect(self, bgr: np.ndarray) -> YOLOEHandleDetection:
        if bgr is None or bgr.size == 0:
            return YOLOEHandleDetection(success=False, message="empty image")

        results = self.model.predict(
            source=bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device or None,
            verbose=False,
        )
        if not results:
            return YOLOEHandleDetection(success=False, message="no yoloe result")

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return YOLOEHandleDetection(success=False, message="no handle detected")

        names = result.names or {}
        img_h, img_w = bgr.shape[:2]
        best_idx = -1
        best_score = -1.0
        best_name = ""
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            cls_name = str(names.get(cls_id, cls_id))
            xyxy = boxes.xyxy[i].cpu().numpy().astype(float)
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]

            if self.prompt_free and not any(k in cls_name.lower() for k in HANDLE_KEYWORDS):
                continue

            score = _score_lever_candidate(cls_name, conf, x1, y1, x2, y2, img_h, img_w)
            if score > best_score:
                best_score = score
                best_idx = i
                best_name = cls_name

        if best_idx < 0 or best_score <= 0.0:
            return YOLOEHandleDetection(success=False, message="no lever handle detected")

        xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(float)
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        cx, cy = _lever_center(x1, y1, x2, y2)
        conf = float(boxes.conf[best_idx].item())
        aspect = _bbox_aspect(x1, y1, x2, y2)

        return YOLOEHandleDetection(
            success=True,
            bbox_xyxy=(x1, y1, x2, y2),
            center_px=(cx, cy),
            confidence=conf,
            class_name=best_name,
            message="yoloe_lever",
            debug={
                "det_count": len(boxes),
                "prompt_free": self.prompt_free,
                "lever_score": best_score,
                "aspect": aspect,
            },
        )


def draw_yoloe_detection(
    bgr: np.ndarray,
    det: YOLOEHandleDetection,
    *,
    extra_lines: Optional[list[str]] = None,
    center_px: Optional[tuple[int, int]] = None,
    cached: bool = False,
    cache_info: str = "",
) -> np.ndarray:
    out = bgr.copy()
    point = center_px or det.center_px
    show = det.success or cached

    if show:
        cx, cy = point
        color = (0, 165, 255) if cached else (0, 0, 255)
        if det.success and not cached:
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"{det.class_name} {det.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.circle(out, (cx, cy), 8, color, 2)
        cv2.circle(out, (cx, cy), 3, color, -1)
        label = "cache" if cached else "center"
        cv2.putText(
            out,
            f"{label} ({cx}, {cy})",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if cached:
        status = "CACHE"
        status_color = (0, 165, 255)
        status_text = cache_info or "cluster from recent detections"
    else:
        status = "OK" if det.success else "FAIL"
        status_color = (0, 255, 0) if det.success else (0, 0, 255)
        status_text = det.message
    cv2.putText(
        out,
        f"{status} {status_text}",
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
