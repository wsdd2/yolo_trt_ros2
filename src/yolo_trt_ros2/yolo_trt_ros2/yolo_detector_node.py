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
        self.prompts = self._parse_prompts(self.get_parameter('prompts').value)

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
        self.declare_parameter(
            'prompts',
            'lever door handle,horizontal door handle,door lever handle,pull door handle',
        )

    def _parse_prompts(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    def _load_class_names(self, class_names_path):
        if not class_names_path:
            return []
        if not os.path.exists(class_names_path):
            self.get_logger().warn('class_names_path does not exist: %s' % class_names_path)
            return []

        class_names = []
        with open(class_names_path, 'r') as names_file:
            for line in names_file:
                name = line.strip()
                if name and not name.startswith('#'):
                    class_names.append(name)
        return class_names

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

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
            label = '%s %.2f' % (class_name, confidence)
            cv2.putText(
                image,
                label,
                (xmin, max(0, ymin - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return image


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
