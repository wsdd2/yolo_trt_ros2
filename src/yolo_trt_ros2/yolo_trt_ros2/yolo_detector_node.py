#!/usr/bin/env python3
import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from detector_msgs.msg import Object2D, Object2DArray
from yolo_trt_ros2.backends.mock_backend import MockBackend


class YoloDetectorNode(Node):
    """Small ROS2 Foxy detector node with a swappable inference backend."""

    def __init__(self):
        super().__init__('yolo_detector')

        self._declare_parameters()
        self.image_topic = self.get_parameter('image_topic').value
        self.objects_topic = self.get_parameter('objects_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.engine_path = self.get_parameter('engine_path').value
        self.model_path = self.get_parameter('model_path').value
        self.model_kind = str(self.get_parameter('model_kind').value).lower()
        self.class_names_path = self.get_parameter('class_names_path').value
        self.yoloe_classes_path = str(self.get_parameter('yoloe_classes_path').value)
        self.input_width = int(self.get_parameter('input_width').value)
        self.input_height = int(self.get_parameter('input_height').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.iou_thres = float(self.get_parameter('iou_thres').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.backend_name = str(self.get_parameter('backend').value).lower()
        self.device = str(self.get_parameter('device').value)
        self.prompt_free = bool(self.get_parameter('prompt_free').value)
        self.best_handle_only = bool(self.get_parameter('best_handle_only').value)
        self.filter_prompt_free_handles = bool(self.get_parameter('filter_prompt_free_handles').value)
        self.mobileclip_path = str(self.get_parameter('mobileclip_path').value)
        self.detect_blue_point = bool(self.get_parameter('detect_blue_point').value)
        self.blue_point_class_name = str(self.get_parameter('blue_point_class_name').value)
        self.blue_point_min_area = float(self.get_parameter('blue_point_min_area').value)
        self.blue_point_max_area_ratio = float(self.get_parameter('blue_point_max_area_ratio').value)
        self.prompts = self._load_yoloe_classes(self.yoloe_classes_path)
        if not self.prompts:
            self.prompts = self._parse_list_value(self.get_parameter('prompts').value)

        self.bridge = CvBridge()
        self.class_names = self._load_class_names(self.class_names_path)
        self.backend = self._create_backend()

        self.objects_pub = self.create_publisher(Object2DArray, self.objects_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            10,
        )

        self.get_logger().info(
            'YOLO detector started: backend=%s, image_topic=%s, objects_topic=%s'
            % (self.backend_name, self.image_topic, self.objects_topic)
        )

    def _declare_parameters(self):
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('objects_topic', '/detector/objects')
        self.declare_parameter('debug_image_topic', '/detector/debug_image')
        self.declare_parameter('engine_path', '')
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_kind', 'auto')
        self.declare_parameter('class_names_path', '')
        self.declare_parameter('yoloe_classes_path', '')
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 640)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('backend', 'mock')
        self.declare_parameter('device', '')
        self.declare_parameter('prompt_free', False)
        self.declare_parameter('best_handle_only', False)
        self.declare_parameter('filter_prompt_free_handles', False)
        self.declare_parameter('mobileclip_path', '')
        self.declare_parameter('detect_blue_point', False)
        self.declare_parameter('blue_point_class_name', 'blue push point')
        self.declare_parameter('blue_point_min_area', 40.0)
        self.declare_parameter('blue_point_max_area_ratio', 0.02)
        self.declare_parameter(
            'prompts',
            'lever door handle,horizontal door handle,door lever handle,pull door handle',
        )

    def _parse_list_value(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    def _load_yoloe_classes(self, path):
        classes = self._load_text_list(path, 'yoloe_classes_path')
        if classes:
            self.get_logger().info('Loaded %d YOLOE classes from %s' % (len(classes), path))
        return classes

    def _load_class_names(self, class_names_path):
        return self._load_text_list(class_names_path, 'class_names_path')

    def _load_text_list(self, path, label):
        if not path:
            return []
        expanded = os.path.expanduser(str(path))
        if not os.path.exists(expanded):
            self.get_logger().warn('%s does not exist: %s' % (label, expanded))
            return []

        values = []
        with open(expanded, 'r', encoding='utf-8') as names_file:
            for line in names_file:
                text = line.strip()
                if not text or text.startswith('#'):
                    continue
                if '#' in text:
                    text = text.split('#', 1)[0].strip()
                for item in text.split(','):
                    name = item.strip()
                    if name:
                        values.append(name)
        return values

    def _create_backend(self):
        if self.backend_name == 'mock':
            return MockBackend(self.class_names)

        if self.backend_name == 'tensorrt':
            # Import only here so mock mode does not need TensorRT installed.
            from yolo_trt_ros2.backends.tensorrt_backend import TensorRTBackend

            return TensorRTBackend(
                engine_path=self.engine_path,
                class_names=self.class_names,
                input_width=self.input_width,
                input_height=self.input_height,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
            )

        if self.backend_name in ('ultralytics', 'yolo', 'yoloe'):
            from yolo_trt_ros2.backends.ultralytics_backend import UltralyticsBackend

            model_kind = self.model_kind
            if self.backend_name in ('yolo', 'yoloe'):
                model_kind = self.backend_name
            return UltralyticsBackend(
                model_path=self.model_path,
                model_kind=model_kind,
                class_names=self.class_names,
                prompts=self.prompts,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                imgsz=self.imgsz,
                device=self.device,
                prompt_free=self.prompt_free,
                best_handle_only=self.best_handle_only,
                mobileclip_path=self.mobileclip_path,
                filter_prompt_free_handles=self.filter_prompt_free_handles,
            )

        raise ValueError('Unsupported backend: %s. Use mock, yoloe, yolo, ultralytics or tensorrt.' % self.backend_name)

    def _image_callback(self, image_msg):
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error('Failed to convert ROS Image to OpenCV: %s' % exc)
            return

        try:
            detections = self.backend.infer(bgr_image)
        except Exception as exc:
            self.get_logger().error('Backend inference failed: %s' % exc)
            return
        if self.detect_blue_point:
            detections = list(detections) + self._detect_blue_point(bgr_image, detections)

        objects_msg = self._build_objects_msg(image_msg.header, detections)
        self.objects_pub.publish(objects_msg)

        if self.publish_debug_image:
            debug_image = self._draw_debug_image(bgr_image.copy(), detections)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            debug_msg.header = image_msg.header
            self.debug_pub.publish(debug_msg)

    def _build_objects_msg(self, header, detections):
        msg = Object2DArray()
        msg.header = header

        for det in detections:
            obj = Object2D()
            obj.class_name = str(det.get('class_name', 'unknown'))
            obj.class_id = int(det.get('class_id', -1))
            obj.confidence = float(det.get('confidence', 0.0))
            obj.xmin = int(det.get('xmin', 0))
            obj.ymin = int(det.get('ymin', 0))
            obj.xmax = int(det.get('xmax', 0))
            obj.ymax = int(det.get('ymax', 0))
            obj.cx = float(det.get('cx', float(obj.xmin + obj.xmax) * 0.5))
            obj.cy = float(det.get('cy', float(obj.ymin + obj.ymax) * 0.5))
            msg.objects.append(obj)

        return msg

    def _draw_debug_image(self, image, detections):
        for det in detections:
            xmin = int(det.get('xmin', 0))
            ymin = int(det.get('ymin', 0))
            xmax = int(det.get('xmax', 0))
            ymax = int(det.get('ymax', 0))
            class_name = str(det.get('class_name', 'unknown'))
            confidence = float(det.get('confidence', 0.0))
            cx = int(round(float(det.get('cx', float(xmin + xmax) * 0.5))))
            cy = int(round(float(det.get('cy', float(ymin + ymax) * 0.5))))

            color = (255, 128, 0) if 'blue' in class_name.lower() else (0, 255, 0)
            center_color = (255, 0, 0) if 'blue' in class_name.lower() else (0, 0, 255)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.circle(image, (cx, cy), 5, center_color, -1)
            label = '%s %.2f' % (class_name, confidence)
            cv2.putText(
                image,
                label,
                (xmin, max(0, ymin - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return image

    def _detect_blue_point(self, image, detections):
        if image is None or image.size == 0:
            return []

        img_h, img_w = image.shape[:2]
        rois = self._blue_search_rois(detections, img_w, img_h)
        candidates = []
        for rx1, ry1, rx2, ry2 in rois:
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            crop = image[ry1:ry2, rx1:rx2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (78, 45, 35), (112, 255, 255))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.blue_point_min_area:
                    continue
                if area > float(img_w * img_h) * max(self.blue_point_max_area_ratio, 1e-6):
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 2 or h <= 2:
                    continue
                aspect = float(w) / float(h)
                if aspect < 0.45 or aspect > 2.2:
                    continue
                perimeter = float(cv2.arcLength(contour, True))
                circularity = 0.0 if perimeter <= 0.0 else 4.0 * 3.14159 * area / (perimeter * perimeter)
                if circularity < 0.25:
                    continue
                moments = cv2.moments(contour)
                if moments['m00'] != 0.0:
                    cx = rx1 + float(moments['m10'] / moments['m00'])
                    cy = ry1 + float(moments['m01'] / moments['m00'])
                else:
                    cx = rx1 + float(x + w) * 0.5
                    cy = ry1 + float(y + h) * 0.5
                candidates.append(
                    {
                        'class_name': self.blue_point_class_name,
                        'class_id': 9001,
                        'confidence': min(0.99, 0.45 + 0.45 * min(1.0, circularity)),
                        'xmin': int(rx1 + x),
                        'ymin': int(ry1 + y),
                        'xmax': int(rx1 + x + w),
                        'ymax': int(ry1 + y + h),
                        'cx': float(cx),
                        'cy': float(cy),
                        'score': area * max(0.2, circularity),
                    }
                )

        if not candidates:
            return []
        best = max(candidates, key=lambda det: det['score'])
        best.pop('score', None)
        return [best]

    def _blue_search_rois(self, detections, img_w, img_h):
        handle_rois = []
        for det in detections:
            name = str(det.get('class_name', '')).lower()
            if not any(token in name for token in ('handle', 'lever', 'pull', 'cabinet door')):
                continue
            x1 = int(det.get('xmin', 0))
            y1 = int(det.get('ymin', 0))
            x2 = int(det.get('xmax', img_w - 1))
            y2 = int(det.get('ymax', img_h - 1))
            pad_x = max(8, int((x2 - x1) * 0.35))
            pad_y = max(8, int((y2 - y1) * 0.25))
            handle_rois.append(
                (
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(img_w, x2 + pad_x),
                    min(img_h, y2 + pad_y),
                )
            )
        return handle_rois or [(0, 0, img_w, img_h)]


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
