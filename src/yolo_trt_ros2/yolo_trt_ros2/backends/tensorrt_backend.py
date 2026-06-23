import os


class TensorRTBackend(object):
    """TensorRT backend placeholder with lazy imports.

    The import of TensorRT/CUDA bindings happens only when this class is
    constructed, so the default mock backend can run on machines without
    JetPack or TensorRT Python packages installed.
    """

    def __init__(self, engine_path, class_names=None, input_width=640, input_height=640,
                 conf_thres=0.25, iou_thres=0.45):
        if not engine_path:
            raise ValueError('engine_path is required when backend=tensorrt')
        if not os.path.exists(engine_path):
            raise RuntimeError('TensorRT engine file does not exist: %s' % engine_path)

        # Lazy imports: keep mock mode independent from TensorRT/CUDA packages.
        try:
            import tensorrt as trt  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                'Failed to import TensorRT Python API. Install TensorRT bindings '
                'for JetPack/L4T R35.3.1, or run with backend=mock.'
            ) from exc

        self.engine_path = engine_path
        self.class_names = class_names or []
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)

        # TODO:
        # - Deserialize the TensorRT engine.
        # - Allocate pagelocked host/device buffers with CUDA Python bindings.
        # - Preprocess BGR images to the engine input layout.
        # - Execute inference and decode YOLO outputs.
        # - Apply NMS and return the unified list[dict] detection format.
        raise NotImplementedError(
            'TensorRT backend interface is reserved but not implemented yet. '
            'Use backend=mock to validate ROS topics first.'
        )

    def infer(self, bgr_image):
        raise NotImplementedError('TensorRT infer is not implemented yet.')
