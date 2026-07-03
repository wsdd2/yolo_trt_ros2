#!/usr/bin/env python3
import math
import os

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

try:
    import tf2_ros
except ImportError:  # pragma: no cover - tf2_ros is provided by ROS.
    tf2_ros = None

from detector_msgs.msg import Object3D, Object3DArray
from detector_msgs.msg import Object2DArray


class CoordinateProjectorNode(Node):
    """Project 2D detector centers into camera-frame 3D coordinates."""

    def __init__(self):
        super().__init__('coordinate_projector')
        self._declare_parameters()

        self.objects_topic = self.get_parameter('objects_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.objects_3d_topic = self.get_parameter('objects_3d_topic').value
        self.target_point_topic = self.get_parameter('target_point_topic').value
        self.target_pose_topic = self.get_parameter('target_pose_topic').value
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.handeye_npy_path = str(self.get_parameter('handeye_npy_path').value)
        self.handeye_target_frame = str(self.get_parameter('handeye_target_frame').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.depth_radius = int(self.get_parameter('depth_radius').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.publish_invalid = bool(self.get_parameter('publish_invalid').value)
        self.publish_target_pose = bool(self.get_parameter('publish_target_pose').value)
        self.stale_depth_sec = float(self.get_parameter('stale_depth_sec').value)
        self.transform_timeout_sec = float(self.get_parameter('transform_timeout_sec').value)
        self.class_filter = set(self._get_string_list_parameter('class_filter'))
        self.pose_orientation_xyzw = self._get_float_list_parameter(
            'target_pose_orientation_xyzw',
            [0.0, 0.0, 0.0, 1.0],
            4,
        )

        self.bridge = CvBridge()
        self.latest_depth_msg = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.handeye_transform = self._load_handeye_transform(self.handeye_npy_path)

        self.tf_buffer = None
        self.tf_listener = None
        if self.target_frame and self.handeye_transform is None:
            if tf2_ros is None:
                self.get_logger().warn('tf2_ros is not available; target_frame transform disabled.')
            else:
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.objects_3d_pub = self.create_publisher(Object3DArray, self.objects_3d_topic, 10)
        self.target_point_pub = self.create_publisher(PointStamped, self.target_point_topic, 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 10)

        self.create_subscription(Image, self.depth_topic, self._depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, 10)
        self.create_subscription(Object2DArray, self.objects_topic, self._objects_callback, 10)

        self.get_logger().info(
            'Coordinate projector started: objects=%s, depth=%s, camera_info=%s, target_frame=%s, handeye=%s'
            % (
                self.objects_topic,
                self.depth_topic,
                self.camera_info_topic,
                self._output_frame_name(),
                self.handeye_npy_path or '<none>',
            )
        )

    def _declare_parameters(self):
        self.declare_parameter('objects_topic', '/detector/objects')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('objects_3d_topic', '/detector/objects_3d')
        self.declare_parameter('target_point_topic', '/detector/target_point')
        self.declare_parameter('target_pose_topic', '/detector/target_pose')
        self.declare_parameter('target_frame', '')
        self.declare_parameter('handeye_npy_path', '')
        self.declare_parameter('handeye_target_frame', 'base_link')
        self.declare_parameter('class_filter', '')
        self.declare_parameter('min_confidence', 0.25)
        self.declare_parameter('depth_radius', 3)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.10)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('publish_invalid', False)
        self.declare_parameter('publish_target_pose', True)
        self.declare_parameter('target_pose_orientation_xyzw', [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('stale_depth_sec', 0.5)
        self.declare_parameter('transform_timeout_sec', 0.05)

    def _get_string_list_parameter(self, name):
        value = self.get_parameter(name).value
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    def _get_float_list_parameter(self, name, default, expected_len):
        value = self.get_parameter(name).value
        if value is None:
            return default
        try:
            values = [float(item) for item in value]
        except TypeError:
            values = default
        if len(values) != expected_len:
            self.get_logger().warn('%s must contain %d values; using default.' % (name, expected_len))
            return default
        return values

    def _load_handeye_transform(self, path):
        if not path:
            return None

        resolved = os.path.expanduser(path)
        if os.path.isdir(resolved):
            candidates = [
                'T_cam2base.npy',
                'T_cam2world.npy',
                'T_camera2base.npy',
                'T_camera2world.npy',
            ]
            for name in candidates:
                candidate = os.path.join(resolved, name)
                if os.path.isfile(candidate):
                    resolved = candidate
                    break

        if not os.path.isfile(resolved):
            self.get_logger().warn('handeye_npy_path does not exist or has no supported transform: %s' % path)
            return None

        try:
            transform = np.load(resolved).astype(np.float64)
        except Exception as exc:
            self.get_logger().warn('Failed to load hand-eye npy: %s' % exc)
            return None

        if transform.shape != (4, 4):
            self.get_logger().warn('hand-eye transform must be 4x4, got shape=%s' % (transform.shape,))
            return None

        self.get_logger().info('Loaded hand-eye camera-to-target transform: %s' % resolved)
        return transform

    def _output_frame_name(self):
        if self.handeye_transform is not None:
            return self.handeye_target_frame or self.target_frame or 'base_link'
        return self.target_frame or '<camera>'

    def _depth_callback(self, depth_msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error('Failed to convert depth image: %s' % exc)
            return

        self.latest_depth_msg = depth_msg
        self.latest_depth = np.asarray(depth)

    def _camera_info_callback(self, camera_info):
        self.latest_camera_info = camera_info

    def _objects_callback(self, objects_msg):
        out_msg = Object3DArray()
        out_msg.header = objects_msg.header

        if self.latest_depth is None or self.latest_camera_info is None:
            out_msg.objects = [self._invalid_object(obj, objects_msg.header, 'missing depth or CameraInfo')
                               for obj in objects_msg.objects if self.publish_invalid]
            self.objects_3d_pub.publish(out_msg)
            return

        if self._depth_is_stale(objects_msg.header):
            self.get_logger().warn('Depth image is stale for detector frame.')

        best_obj = None
        for obj in objects_msg.objects:
            if not self._passes_filter(obj):
                continue

            object_3d = self._project_object(obj, objects_msg.header)
            if object_3d.valid or self.publish_invalid:
                out_msg.objects.append(object_3d)
            if object_3d.valid and (best_obj is None or obj.confidence > best_obj.detection.confidence):
                best_obj = object_3d

        self.objects_3d_pub.publish(out_msg)
        if best_obj is not None:
            self._publish_target(best_obj)

    def _passes_filter(self, obj):
        if obj.confidence < self.min_confidence:
            return False
        if self.class_filter and obj.class_name not in self.class_filter:
            return False
        return True

    def _depth_is_stale(self, objects_header):
        if self.stale_depth_sec <= 0.0 or self.latest_depth_msg is None:
            return False
        object_t = self._stamp_to_sec(objects_header.stamp)
        depth_t = self._stamp_to_sec(self.latest_depth_msg.header.stamp)
        if object_t <= 0.0 or depth_t <= 0.0:
            return False
        return abs(object_t - depth_t) > self.stale_depth_sec

    def _stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _project_object(self, obj, header):
        depth_m = self._sample_depth_m(obj.cx, obj.cy)
        if depth_m is None:
            return self._invalid_object(obj, header, 'no valid depth near detection center')

        try:
            point_camera = self._deproject(float(obj.cx), float(obj.cy), depth_m)
        except RuntimeError as exc:
            return self._invalid_object(obj, header, str(exc))
        point_target = point_camera
        target_frame = header.frame_id
        message = 'camera_frame'

        if self.handeye_transform is not None:
            point_target = self._apply_transform(self.handeye_transform, point_camera)
            target_frame = self.handeye_target_frame or self.target_frame or 'base_link'
            message = 'handeye_npy'
        elif self.target_frame:
            transformed = self._transform_point(point_camera, header.frame_id, self.target_frame)
            if transformed is None:
                return self._invalid_object(obj, header, 'TF transform unavailable')
            point_target = transformed
            target_frame = self.target_frame
            message = 'transformed'

        msg = Object3D()
        msg.header = header
        msg.detection = obj
        msg.valid = True
        msg.cached = False
        msg.source_frame = header.frame_id
        msg.target_frame = target_frame
        msg.depth_m = float(depth_m)
        msg.point_camera = self._point_to_msg(point_camera)
        msg.point_target = self._point_to_msg(point_target)
        msg.message = message
        return msg

    def _invalid_object(self, obj, header, message):
        msg = Object3D()
        msg.header = header
        msg.detection = obj
        msg.valid = False
        msg.cached = False
        msg.source_frame = header.frame_id
        if self.handeye_transform is not None:
            msg.target_frame = self._output_frame_name()
        else:
            msg.target_frame = self.target_frame or header.frame_id
        msg.depth_m = 0.0
        msg.point_camera = Point()
        msg.point_target = Point()
        msg.message = message
        return msg

    def _sample_depth_m(self, u, v):
        depth = self.latest_depth
        if depth is None or depth.size == 0:
            return None

        h, w = depth.shape[:2]
        u_depth, v_depth = self._map_pixel_to_depth_image(u, v, w, h)
        x0 = max(0, u_depth - self.depth_radius)
        x1 = min(w, u_depth + self.depth_radius + 1)
        y0 = max(0, v_depth - self.depth_radius)
        y1 = min(h, v_depth + self.depth_radius + 1)
        patch = np.asarray(depth[y0:y1, x0:x1])
        if patch.size == 0:
            return None

        depth_m = self._depth_patch_to_meters(patch)
        valid = depth_m[
            np.isfinite(depth_m)
            & (depth_m >= self.min_depth_m)
            & (depth_m <= self.max_depth_m)
        ]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _map_pixel_to_depth_image(self, u, v, depth_w, depth_h):
        info = self.latest_camera_info
        image_w = int(info.width) if info is not None and info.width else depth_w
        image_h = int(info.height) if info is not None and info.height else depth_h
        if image_w <= 0 or image_h <= 0:
            image_w, image_h = depth_w, depth_h
        x = int(round(float(u) * float(depth_w) / float(image_w)))
        y = int(round(float(v) * float(depth_h) / float(image_h)))
        return max(0, min(depth_w - 1, x)), max(0, min(depth_h - 1, y))

    def _depth_patch_to_meters(self, patch):
        if np.issubdtype(patch.dtype, np.integer):
            return patch.astype(np.float64) * self.depth_scale
        return patch.astype(np.float64)

    def _deproject(self, u, v, depth_m):
        k = self.latest_camera_info.k
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        if fx == 0.0 or fy == 0.0:
            raise RuntimeError('CameraInfo contains invalid focal length.')
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        return np.array([x, y, depth_m], dtype=np.float64)

    def _transform_point(self, point, source_frame, target_frame):
        if self.tf_buffer is None:
            return None
        if not source_frame:
            self.get_logger().warn('Object header has empty frame_id; cannot transform.')
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.transform_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn('TF lookup failed: %s' % exc)
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        rotated = self._rotate_vector(point, [q.x, q.y, q.z, q.w])
        return rotated + np.array([t.x, t.y, t.z], dtype=np.float64)

    def _apply_transform(self, transform, point):
        point_h = np.ones(4, dtype=np.float64)
        point_h[:3] = np.asarray(point, dtype=np.float64).reshape(3)
        return (transform @ point_h)[:3]

    def _rotate_vector(self, vector, quat_xyzw):
        x, y, z, w = [float(v) for v in quat_xyzw]
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            return np.asarray(vector, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        qvec = np.array([x, y, z], dtype=np.float64)
        vec = np.asarray(vector, dtype=np.float64)
        uv = np.cross(qvec, vec)
        uuv = np.cross(qvec, uv)
        return vec + 2.0 * (w * uv + uuv)

    def _publish_target(self, object_3d):
        header = object_3d.header
        header.frame_id = object_3d.target_frame or object_3d.source_frame

        point_msg = PointStamped()
        point_msg.header = header
        point_msg.point = object_3d.point_target
        self.target_point_pub.publish(point_msg)

        if self.publish_target_pose:
            pose_msg = PoseStamped()
            pose_msg.header = header
            pose_msg.pose.position = object_3d.point_target
            qx, qy, qz, qw = self.pose_orientation_xyzw
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            self.target_pose_pub.publish(pose_msg)

    def _point_to_msg(self, point):
        msg = Point()
        msg.x = float(point[0])
        msg.y = float(point[1])
        msg.z = float(point[2])
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CoordinateProjectorNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
