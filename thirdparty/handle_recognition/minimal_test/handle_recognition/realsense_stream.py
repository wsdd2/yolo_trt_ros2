# -*- coding: utf-8 -*-
"""Minimal RealSense RGB-D stream via pyrealsense2 (aligned depth to color)."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    rs = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class RealSenseRGBD:
    """Fetch synchronized color (RGB) and depth frames aligned to color."""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        depth_width: int = 640,
        depth_height: int = 480,
        depth_fps: Optional[int] = None,
        serial: str = "",
        index: int = 0,
    ) -> None:
        if rs is None:
            raise RuntimeError("未安装 pyrealsense2，请运行: pip install pyrealsense2") from _IMPORT_ERROR
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps if depth_fps is not None else fps
        self.serial = serial.strip()
        self.index = index
        self.depth_scale = 1.0
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._profile = None
        self._started = False
        self.active_config: dict[str, int] = {}

    @staticmethod
    def list_devices() -> list[dict]:
        if rs is None:
            return []
        out = []
        for i, dev in enumerate(rs.context().query_devices()):
            serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else ""
            name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "RealSense"
            out.append({"index": i, "serial": serial, "model": name})
        return out

    @staticmethod
    def list_video_streams() -> list[dict]:
        if rs is None:
            return []
        out = []
        for dev_idx, dev in enumerate(rs.context().query_devices()):
            serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else ""
            name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "RealSense"
            for sensor in dev.query_sensors():
                sensor_name = sensor.get_info(rs.camera_info.name) if sensor.supports(rs.camera_info.name) else "sensor"
                for profile in sensor.get_stream_profiles():
                    try:
                        vp = profile.as_video_stream_profile()
                    except RuntimeError:
                        continue
                    intr = vp.get_intrinsics()
                    out.append(
                        {
                            "device_index": dev_idx,
                            "serial": serial,
                            "model": name,
                            "sensor": sensor_name,
                            "stream": str(profile.stream_type()).split(".")[-1],
                            "format": str(profile.format()).split(".")[-1],
                            "width": int(intr.width),
                            "height": int(intr.height),
                            "fps": int(profile.fps()),
                        }
                    )
        return out

    @staticmethod
    def reset_device(serial: str = "", index: int = 0) -> bool:
        if rs is None:
            return False
        devices = list(rs.context().query_devices())
        if not devices:
            return False
        target = None
        serial = serial.strip()
        if serial:
            for dev in devices:
                dev_serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else ""
                if dev_serial == serial:
                    target = dev
                    break
        elif 0 <= index < len(devices):
            target = devices[index]
        if target is None:
            return False
        target.hardware_reset()
        return True

    def _resolve_serial(self) -> str:
        if self.serial:
            return self.serial
        devices = self.list_devices()
        if devices and self.index < len(devices):
            return str(devices[self.index].get("serial") or "")
        return ""

    def start(self) -> None:
        if self._started:
            return
        serial = self._resolve_serial()
        attempts = [
            (self.width, self.height, self.fps, self.depth_width, self.depth_height, self.depth_fps),
        ]
        same_size = (self.width, self.height, self.fps, self.width, self.height, self.fps)
        if same_size not in attempts:
            attempts.append(same_size)
        for fallback in ((640, 480, 30, 640, 480, 30), (640, 480, 15, 640, 480, 15)):
            if fallback not in attempts:
                attempts.append(fallback)

        last_error: Optional[RuntimeError] = None
        tried = []
        for color_w, color_h, color_fps, depth_w, depth_h, depth_fps in attempts:
            config = rs.config()
            if serial:
                config.enable_device(serial)
            config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)
            config.enable_stream(rs.stream.color, color_w, color_h, rs.format.rgb8, color_fps)

            self._pipeline = rs.pipeline()
            tried.append(f"color={color_w}x{color_h}@{color_fps}, depth={depth_w}x{depth_h}@{depth_fps}")
            try:
                self._profile = self._pipeline.start(config)
            except RuntimeError as exc:
                last_error = exc
                self._pipeline = None
                continue
            self.width = color_w
            self.height = color_h
            self.fps = color_fps
            self.depth_width = depth_w
            self.depth_height = depth_h
            self.depth_fps = depth_fps
            self.active_config = {
                "color_width": color_w,
                "color_height": color_h,
                "color_fps": color_fps,
                "depth_width": depth_w,
                "depth_height": depth_h,
                "depth_fps": depth_fps,
            }
            break

        if self._profile is None:
            detail = "; ".join(tried)
            if last_error is not None:
                raise RuntimeError(f"{last_error}; tried: {detail}") from last_error
            raise RuntimeError(f"RealSense 启动失败; tried: {detail}")

        self._align = rs.align(rs.stream.color)
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self._started = True

    def stop(self) -> None:
        if self._pipeline is not None and self._started:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
        self._started = False
        self._profile = None

    def close(self) -> None:
        self.stop()
        self._pipeline = None
        self._align = None

    def restart(self) -> None:
        self.close()
        self._profile = None
        self.start()

    def color_intrinsics(self) -> tuple[np.ndarray, np.ndarray, dict]:
        if self._profile is None:
            raise RuntimeError("RealSense 尚未启动")
        color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        camera_matrix = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        coeffs = np.asarray(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)
        info = {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "coeffs": coeffs.reshape(-1).astype(float).tolist(),
            "depth_scale": float(self.depth_scale),
        }
        return camera_matrix, coeffs, info

    def fetch(self, timeout_ms: int = 3000) -> Optional[dict[str, np.ndarray]]:
        if not self._started or self._pipeline is None or self._align is None:
            raise RuntimeError("RealSense 未启动，请先调用 start()")

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError:
            return None
        if frames is None:
            return None

        frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        elif rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        raw_depth = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
        depth_mm = raw_depth.astype(np.float32) * self.depth_scale * 1000.0
        depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

        return {
            "rgb": np.ascontiguousarray(rgb),
            "depth": np.ascontiguousarray(depth_mm),
        }

    def __enter__(self) -> "RealSenseRGBD":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
