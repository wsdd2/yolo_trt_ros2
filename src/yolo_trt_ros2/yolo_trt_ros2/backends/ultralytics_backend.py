import os


class UltralyticsBackend(object):
    """Ultralytics YOLO/YOLOE backend with lazy imports for ROS2 Foxy."""

    HANDLE_KEYWORDS = ('lever', 'pull', 'handle')
    LOCK_KEYWORDS = ('lock', 'keyhole', 'key', 'deadbolt', 'cylinder', 'knob')

    def __init__(
        self,
        model_path,
        model_kind='auto',
        class_names=None,
        prompts=None,
        conf_thres=0.25,
        iou_thres=0.45,
        imgsz=640,
        device='',
        prompt_free=False,
        best_handle_only=False,
        mobileclip_path='',
        filter_prompt_free_handles=False,
    ):
        if not model_path:
            raise ValueError('model_path is required when backend=ultralytics/yolo/yoloe')

        try:
            from ultralytics import YOLO, YOLOE
        except ImportError as exc:
            raise RuntimeError(
                'Failed to import ultralytics. Install it in the ROS Python environment, '
                'or use backend=mock.'
            ) from exc

        self.model_path = model_path
        self.model_kind = self._resolve_model_kind(model_kind, model_path)
        self.class_names = class_names or []
        self.prompts = [p.strip() for p in (prompts or []) if p.strip()]
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        if isinstance(imgsz, (list, tuple)):
            if len(imgsz) != 2:
                raise ValueError('imgsz sequence must be [height, width]')
            self.imgsz = [int(imgsz[0]), int(imgsz[1])]
        else:
            self.imgsz = int(imgsz)
        self.device = device
        self.prompt_free = bool(prompt_free)
        self.best_handle_only = bool(best_handle_only)
        self.mobileclip_path = str(mobileclip_path or '')
        self.filter_prompt_free_handles = bool(filter_prompt_free_handles)

        if self.model_kind == 'yoloe':
            self._patch_mobileclip_asset()
            self.model = YOLOE(model_path)
            if not self.prompt_free and self.prompts:
                self.model.set_classes(self.prompts)
        else:
            self.model = YOLO(model_path)

    def _resolve_model_kind(self, model_kind, model_path):
        kind = str(model_kind or 'auto').lower()
        if kind in ('yolo', 'yoloe'):
            return kind
        return 'yoloe' if 'yoloe' in str(model_path).lower() else 'yolo'

    def _patch_mobileclip_asset(self):
        if not self.mobileclip_path:
            return

        resolved = os.path.expanduser(self.mobileclip_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError('mobileclip_path does not exist: %s' % resolved)

        try:
            from ultralytics.nn import text_model
            from ultralytics.utils import downloads
        except Exception:
            return

        original = downloads.attempt_download_asset

        def _attempt_download_asset(asset, *args, **kwargs):
            name = str(asset).lower()
            if 'mobileclip' in name or 'blt' in name:
                return resolved
            return original(asset, *args, **kwargs)

        # Ultralytics releases differ here: some keep a module-level alias in
        # nn.text_model, while others import from utils.downloads inside the
        # MobileCLIP constructor. Patch both entry points so mobileclip_path is
        # honored consistently.
        text_model.attempt_download_asset = _attempt_download_asset
        downloads.attempt_download_asset = _attempt_download_asset

    def infer(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            return []

        results = self.model.predict(
            source=bgr_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            device=self.device or None,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names or {}
        detections = []
        img_h, img_w = bgr_image.shape[:2]
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            class_name = str(names.get(cls_id, self._class_name(cls_id)))
            xyxy = boxes.xyxy[i].cpu().numpy().astype(float)
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(0, min(img_w - 1, x2))
            y2 = max(0, min(img_h - 1, y2))

            if self.prompt_free and self.model_kind == 'yoloe' and self.filter_prompt_free_handles:
                if not any(k in class_name.lower() for k in self.HANDLE_KEYWORDS):
                    continue

            detections.append(
                {
                    'class_name': class_name,
                    'class_id': cls_id,
                    'confidence': conf,
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'score': self._handle_score(class_name, conf, x1, y1, x2, y2, img_h),
                }
            )

        if self.best_handle_only:
            detections = [d for d in detections if d['score'] > 0.0]
            if not detections:
                return []
            best = max(detections, key=lambda d: d['score'])
            return [self._with_lever_center(best)]

        return [self._strip_score(d) for d in detections]

    def _class_name(self, class_id):
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def _strip_score(self, det):
        det = dict(det)
        det.pop('score', None)
        return det

    def _with_lever_center(self, det):
        det = self._strip_score(det)
        x1, y1, x2, y2 = det['xmin'], det['ymin'], det['xmax'], det['ymax']
        det['cx'] = int(round(x1 + (x2 - x1) * 0.55))
        det['cy'] = int(round(y1 + (y2 - y1) * 0.50))
        return det

    def _handle_score(self, class_name, conf, x1, y1, x2, y2, img_h):
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if w < 16 or h < 8:
            return 0.0

        name = str(class_name).lower()
        if any(k in name for k in self.LOCK_KEYWORDS):
            return 0.0

        aspect = float(w) / float(h)
        score = float(conf)
        if aspect < 1.15:
            score *= 0.15
        elif aspect < 1.35:
            score *= 0.45
        else:
            score *= 1.0 + min(0.8, (aspect - 1.35) * 0.25)

        if w < h * 1.1:
            score *= 0.2
        if 'lever' in name or 'pull' in name:
            score *= 1.4
        elif 'handle' in name:
            score *= 1.15
        elif 'knob' in name:
            score *= 0.25

        if aspect >= 1.35:
            cy = (y1 + y2) * 0.5
            score *= 1.0 + 0.25 * (1.0 - min(1.0, cy / max(float(img_h), 1.0)))

        return score
