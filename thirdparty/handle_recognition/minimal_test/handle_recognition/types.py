# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class HandleDetection:
    """Detection result for a lever-style door handle."""

    success: bool
    grasp_point: tuple[int, int] = (0, 0)
    handle_contour: Optional[np.ndarray] = None
    backplate_contour: Optional[np.ndarray] = None
    lever_contour: Optional[np.ndarray] = None
    lever_axis: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    backplate_center: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    message: str = ""
    debug: dict[str, Any] = field(default_factory=dict)
