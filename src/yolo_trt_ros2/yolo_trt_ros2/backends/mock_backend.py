class MockBackend(object):
    """Deterministic fake detector used to verify the ROS image pipeline."""

    def __init__(self, class_names=None):
        self.class_names = class_names or [
            'cabinet_door',
            'door_frame',
            'button',
            'toggle_switch',
            'knob',
        ]

    def infer(self, bgr_image):
        height, width = bgr_image.shape[:2]
        box_w = max(40, int(width * 0.28))
        box_h = max(40, int(height * 0.24))
        xmin = max(0, int(width * 0.36))
        ymin = max(0, int(height * 0.34))
        xmax = min(width - 1, xmin + box_w)
        ymax = min(height - 1, ymin + box_h)

        class_name = self.class_names[0] if self.class_names else 'cabinet_door'

        return [
            {
                'class_name': class_name,
                'class_id': 0,
                'confidence': 0.90,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
            }
        ]
