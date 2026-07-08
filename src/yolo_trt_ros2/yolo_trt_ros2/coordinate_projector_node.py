#!/usr/bin/env python3
import json
import math
import os
import sys
import time
import ctypes
from pathlib import Path


def _bootstrap_h2_python_paths():
    # H2 daily launch runs from system Python. Keep these paths local to this
    # node so console-script metadata issues and stripped PYTHONPATH do not
    # break rclpy or Unitree SDK imports.
    for path in (
        '/opt/ros/humble/local/lib/python3.10/dist-packages',
        '/opt/ros/humble/lib/python3.10/site-packages',
        '/home/unitree/MscapeTech/Foxy_ROS/install/detector_msgs/local/lib/python3.10/dist-packages',
        '/home/unitree/MscapeTech/Foxy_ROS/install/detector_msgs/lib/python3.10/site-packages',
        '/home/unitree/MscapeTech/unitree_sdk2_python',
    ):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    ros_lib_dirs = []
    ros_lib_root = '/opt/ros/humble/lib'
    if os.path.isdir(ros_lib_root):
        ros_lib_dirs.append(ros_lib_root)
        for root, _dirs, files in os.walk(ros_lib_root):
            if 'librcl_action.so' in files and root not in ros_lib_dirs:
                ros_lib_dirs.append(root)

    old_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    old_ld_parts = [part for part in old_ld_library_path.split(':') if part]
    missing_ld_dirs = [path for path in ros_lib_dirs if path not in old_ld_parts]
    if missing_ld_dirs:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(missing_ld_dirs + old_ld_parts)
        # The dynamic loader reads LD_LIBRARY_PATH at process startup. If this
        # process was spawned with a stripped environment, restart once so
        # rclpy's C extension can resolve librcl_action.so and friends.
        if os.environ.get('YOLO_TRT_ROS2_LD_BOOTSTRAPPED') != '1':
            os.environ['YOLO_TRT_ROS2_LD_BOOTSTRAPPED'] = '1'
            os.execv(sys.executable, [sys.executable] + sys.argv)

    for library in (
        'librcl_action.so',
        'librcl.so',
        'librmw.so',
        'librcutils.so',
    ):
        for directory in ros_lib_dirs:
            path = os.path.join(directory, library)
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    break
                except OSError:
                    pass


_bootstrap_h2_python_paths()

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

try:
    import tf2_ros
except ImportError:  # pragma: no cover - tf2_ros is provided by ROS.
    tf2_ros = None

from detector_msgs.msg import Object3D, Object3DArray, RobotInspectionStatus
from detector_msgs.msg import Object2DArray


G1_H2_JOINT_INDEX = {
    'waist_yaw_joint': 12,
    'waist_roll_joint': 13,
    'waist_pitch_joint': 14,
    'left_shoulder_pitch_joint': 15,
    'left_shoulder_roll_joint': 16,
    'left_shoulder_yaw_joint': 17,
    'left_elbow_joint': 18,
    'left_wrist_roll_joint': 19,
    'left_wrist_pitch_joint': 20,
    'left_wrist_yaw_joint': 21,
    'right_shoulder_pitch_joint': 22,
    'right_shoulder_roll_joint': 23,
    'right_shoulder_yaw_joint': 24,
    'right_elbow_joint': 25,
    'right_wrist_roll_joint': 26,
    'right_wrist_pitch_joint': 27,
    'right_wrist_yaw_joint': 28,
}

H2_XR_DUAL_ARM_JOINTS = [
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]


class UnitreeLowStateJointReader:
    """Tiny Unitree lowstate reader used only when eye-in-hand FK is enabled."""

    def __init__(self, network_interface='', domain_id=0, lowstate_topic='rt/lowstate'):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        self._latest = None
        self.topic = lowstate_topic or 'rt/lowstate'
        if network_interface:
            ChannelFactoryInitialize(int(domain_id), str(network_interface))
        else:
            ChannelFactoryInitialize(int(domain_id))
        self._subscriber = ChannelSubscriber(self.topic, LowState_)
        self._subscriber.Init(self._on_lowstate, 10)

    def _on_lowstate(self, msg):
        self._latest = msg

    def wait(self, timeout_sec=2.0):
        deadline = time.time() + float(timeout_sec)
        while self._latest is None and time.time() < deadline:
            time.sleep(0.02)
        return self._latest is not None

    def joint_positions_by_name(self):
        if self._latest is None:
            return {}
        return {
            name: float(self._latest.motor_state[index].q)
            for name, index in G1_H2_JOINT_INDEX.items()
        }


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
        self.target_joint_state_topic = self.get_parameter('target_joint_state_topic').value
        self.current_joint_state_topic = self.get_parameter('current_joint_state_topic').value
        self.objects_ik_topic = self.get_parameter('objects_ik_topic').value
        self.robot_status_topic = self.get_parameter('robot_status_topic').value
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.handeye_npy_path = str(self.get_parameter('handeye_npy_path').value)
        self.handeye_mode = str(self.get_parameter('handeye_mode').value).strip().lower().replace('_', '-')
        self.handeye_target_frame = str(self.get_parameter('handeye_target_frame').value)
        self.robot_model = str(self.get_parameter('robot_model').value).strip().lower()
        self.workspace_root = str(self.get_parameter('workspace_root').value)
        self.urdf_path = str(self.get_parameter('urdf_path').value)
        self.base_link = str(self.get_parameter('base_link').value)
        self.hand_link = str(self.get_parameter('hand_link').value)
        self.camera_link = str(self.get_parameter('camera_link').value)
        self.network_interface = str(self.get_parameter('network_interface').value)
        self.domain_id = int(self.get_parameter('domain_id').value)
        self.lowstate_topic = str(self.get_parameter('lowstate_topic').value)
        self.joint_json_path = str(self.get_parameter('joint_json_path').value)
        self.lock_waist = bool(self.get_parameter('lock_waist').value)
        self.h2_ee_offset_xyz = self._get_float_list_parameter('h2_ee_offset_xyz', [0.05, 0.0, 0.0], 3)
        self.fk_backend_requested = str(self.get_parameter('fk_backend').value).strip().lower()
        self.fk_backend_active = 'none'
        self.h2_xr_scripts_dir = str(self.get_parameter('h2_xr_scripts_dir').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.depth_radius = int(self.get_parameter('depth_radius').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.publish_invalid = bool(self.get_parameter('publish_invalid').value)
        self.publish_target_pose = bool(self.get_parameter('publish_target_pose').value)
        self.publish_target_joint_state = bool(self.get_parameter('publish_target_joint_state').value)
        self.publish_current_joint_state = bool(self.get_parameter('publish_current_joint_state').value)
        self.publish_objects_ik_json = bool(self.get_parameter('publish_objects_ik_json').value)
        self.publish_failed_ik_solution = bool(self.get_parameter('publish_failed_ik_solution').value)
        self.stale_depth_sec = float(self.get_parameter('stale_depth_sec').value)
        self.transform_timeout_sec = float(self.get_parameter('transform_timeout_sec').value)
        self.class_filter = set(self._get_string_list_parameter('class_filter'))
        self.pose_orientation_xyzw = self._get_float_list_parameter(
            'target_pose_orientation_xyzw',
            [0.0, 0.0, 0.0, 1.0],
            4,
        )
        self.ik_target_link = str(self.get_parameter('ik_target_link').value)
        self.ik_active_joints = self._get_string_list_parameter('ik_active_joints')
        self.ik_end_effector_offset_xyz = self._get_float_list_parameter('ik_end_effector_offset_xyz', [0.0, 0.0, 0.0], 3)
        self.ik_target_position_offset_xyz = self._get_float_list_parameter('ik_target_position_offset_xyz', [0.0, 0.0, 0.0], 3)
        self.ik_max_iterations = int(self.get_parameter('ik_max_iterations').value)
        self.ik_tolerance_position = float(self.get_parameter('ik_tolerance_position').value)
        self.ik_tolerance_orientation = float(self.get_parameter('ik_tolerance_orientation').value)
        self.ik_damping = float(self.get_parameter('ik_damping').value)
        self.ik_step_scale = float(self.get_parameter('ik_step_scale').value)
        self.ik_position_weight = float(self.get_parameter('ik_position_weight').value)
        self.ik_orientation_weight = float(self.get_parameter('ik_orientation_weight').value)
        self.max_ik_objects = int(self.get_parameter('max_ik_objects').value)

        self.bridge = CvBridge()
        self.latest_depth_msg = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.handeye_transform = None
        self.T_cam2hand = None
        self._fk_model = None
        self._xr_ik = None
        self._xr_current_ee_poses = None
        self._ik_solver = None
        self._joint_reader = None
        self._fk_error = ''
        self._ik_error = ''
        self._last_ik_warning_sec = 0.0
        self.latest_robot_status = None
        self._configure_handeye_and_fk()

        self.tf_buffer = None
        self.tf_listener = None
        if self.target_frame and self.handeye_transform is None and self.T_cam2hand is None:
            if tf2_ros is None:
                self.get_logger().warn('tf2_ros is not available; target_frame transform disabled.')
            else:
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.objects_3d_pub = self.create_publisher(Object3DArray, self.objects_3d_topic, 10)
        self.target_point_pub = self.create_publisher(PointStamped, self.target_point_topic, 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, self.target_pose_topic, 10)
        self.target_joint_state_pub = self.create_publisher(JointState, self.target_joint_state_topic, 10)
        self.current_joint_state_pub = self.create_publisher(JointState, self.current_joint_state_topic, 10)
        self.objects_ik_pub = self.create_publisher(String, self.objects_ik_topic, 10)

        self.create_subscription(Image, self.depth_topic, self._depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, 10)
        self.create_subscription(Object2DArray, self.objects_topic, self._objects_callback, 10)
        self.create_subscription(RobotInspectionStatus, self.robot_status_topic, self._robot_status_callback, 10)
        self.create_timer(0.1, self._current_joint_timer_callback)

        self.get_logger().info(
            'Coordinate projector started: objects=%s, depth=%s, camera_info=%s, target_frame=%s, handeye_mode=%s, handeye=%s'
            % (
                self.objects_topic,
                self.depth_topic,
                self.camera_info_topic,
                self._output_frame_name(),
                self.handeye_mode or '<none>',
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
        self.declare_parameter('target_joint_state_topic', '/detector/target_joint_state')
        self.declare_parameter('current_joint_state_topic', '/detector/current_joint_state')
        self.declare_parameter('objects_ik_topic', '/detector/objects_ik_json')
        self.declare_parameter('robot_status_topic', '/robot/inspection_status')
        self.declare_parameter('target_frame', '')
        self.declare_parameter('handeye_npy_path', '')
        self.declare_parameter('handeye_mode', 'eye-to-hand')
        self.declare_parameter('handeye_target_frame', 'base_link')
        self.declare_parameter('robot_model', 'h2')
        self.declare_parameter('workspace_root', '/home/unitree/MscapeTech')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('base_link', '')
        self.declare_parameter('hand_link', '')
        self.declare_parameter('camera_link', '')
        self.declare_parameter('network_interface', '')
        self.declare_parameter('domain_id', 0)
        self.declare_parameter('lowstate_topic', 'rt/lowstate')
        self.declare_parameter('joint_json_path', '')
        self.declare_parameter('lock_waist', True)
        self.declare_parameter('h2_ee_offset_xyz', [0.05, 0.0, 0.0])
        self.declare_parameter('fk_backend', 'auto')
        self.declare_parameter('h2_xr_scripts_dir', '/home/unitree/H2_joint_cartesian/scripts')
        self.declare_parameter('class_filter', '')
        self.declare_parameter('min_confidence', 0.25)
        self.declare_parameter('depth_radius', 3)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.10)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('publish_invalid', False)
        self.declare_parameter('publish_target_pose', True)
        self.declare_parameter('publish_target_joint_state', False)
        self.declare_parameter('publish_current_joint_state', True)
        self.declare_parameter('publish_objects_ik_json', True)
        self.declare_parameter('publish_failed_ik_solution', False)
        self.declare_parameter('target_pose_orientation_xyzw', [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('ik_target_link', '')
        self.declare_parameter('ik_active_joints', '')
        self.declare_parameter('ik_end_effector_offset_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('ik_target_position_offset_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('ik_max_iterations', 120)
        self.declare_parameter('ik_tolerance_position', 0.005)
        self.declare_parameter('ik_tolerance_orientation', 3.2)
        self.declare_parameter('ik_damping', 0.001)
        self.declare_parameter('ik_step_scale', 0.4)
        self.declare_parameter('ik_position_weight', 1.0)
        self.declare_parameter('ik_orientation_weight', 0.0)
        self.declare_parameter('max_ik_objects', 24)
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

    def _configure_handeye_and_fk(self):
        if not self.handeye_npy_path:
            return

        if self.handeye_mode in ('eye-in-hand', 'eyeinhand'):
            self.T_cam2hand = self._load_eye_in_hand_transform(self.handeye_npy_path)
            if self.T_cam2hand is not None:
                self._configure_fk()
            return

        self.handeye_transform = self._load_handeye_transform(self.handeye_npy_path)
        if self.handeye_transform is not None and (self.publish_target_joint_state or self.publish_objects_ik_json):
            self._configure_fk()

    def _workspace_path(self, *parts):
        return Path(os.path.expanduser(self.workspace_root)).joinpath(*parts)

    def _resolve_urdf_path(self):
        if self.urdf_path:
            return Path(os.path.expanduser(self.urdf_path))
        if self.robot_model == 'h2':
            return self._workspace_path('unitree_ros', 'robots', 'h2_description', 'H2.urdf')
        return self._workspace_path('unitree_ros', 'robots', 'g1_description', 'g1_29dof_rev_1_0.urdf')

    def _default_base_link(self):
        if self.base_link:
            return self.base_link
        if self.robot_model == 'h2':
            return 'torso_link'
        return 'pelvis'

    def _default_hand_link(self):
        if self.hand_link:
            return self.hand_link
        return 'right_wrist_yaw_link'

    def _configure_fk(self):
        urdf = self._resolve_urdf_path()
        if not urdf.is_file() and self.fk_backend_requested in ('auto', 'urdf'):
            self._fk_error = 'URDF does not exist: %s' % urdf
            self.get_logger().warn(self._fk_error)
            if self.fk_backend_requested == 'urdf':
                return

        try:
            if urdf.is_file():
                workspace = Path(os.path.expanduser(self.workspace_root))
                robot_kinematics_dir = workspace / 'robot_kinematics'
                joint_to_pose_dir = robot_kinematics_dir / 'joint_to_pose'
                pose_to_joint_dir = robot_kinematics_dir / 'pose_to_joint'
                for path in (robot_kinematics_dir, joint_to_pose_dir, pose_to_joint_dir):
                    text = str(path)
                    if text not in sys.path:
                        sys.path.insert(0, text)
                from fk_urdf import URDFFK

                self._fk_model = URDFFK(str(urdf))
                self.urdf_path = str(urdf)
            self.base_link = self._default_base_link()
            self.hand_link = self._default_hand_link()
            self.get_logger().info(
                'Eye-in-hand FK enabled: robot=%s, backend=%s, urdf=%s, base=%s, hand=%s'
                % (self.robot_model, self.fk_backend_requested, self.urdf_path or '<none>', self.base_link, self.hand_link)
            )
        except Exception as exc:
            self._fk_error = 'Failed to initialize FK: %s' % exc
            self.get_logger().warn(self._fk_error)
            if self.fk_backend_requested == 'urdf':
                return

        if self.publish_target_joint_state or self.publish_objects_ik_json:
            try:
                from ik_urdf import URDFIK

                self._ik_solver = URDFIK(str(urdf))
                if not self.ik_target_link:
                    self.ik_target_link = self.hand_link
                self.get_logger().info(
                    'IK target joint publisher enabled: topic=%s, base=%s, target_link=%s, active_joints=%s'
                    % (
                        self.target_joint_state_topic,
                        self.base_link,
                        self.ik_target_link,
                        ','.join(self.ik_active_joints) if self.ik_active_joints else '<chain-default>',
                    )
                )
            except Exception as exc:
                self._ik_error = 'Failed to initialize IK: %s' % exc
                self.get_logger().warn(self._ik_error)

        if self.joint_json_path:
            return

        try:
            self._joint_reader = UnitreeLowStateJointReader(
                self.network_interface,
                self.domain_id,
                self.lowstate_topic,
            )
            if not self._joint_reader.wait(timeout_sec=2.0):
                self.get_logger().warn(
                    'No %s received yet; point_target will stay camera-only until joints arrive.'
                    % self.lowstate_topic
                )
        except Exception as exc:
            self._fk_error = 'Failed to initialize Unitree lowstate reader: %s' % exc
            self.get_logger().warn(self._fk_error)

    def _ensure_xr_fk(self):
        if self._xr_ik is not None and self._xr_current_ee_poses is not None:
            return
        if self.base_link != 'pelvis':
            raise RuntimeError('xr_pinocchio FK only supports base_link=pelvis')
        candidates = [
            Path(os.path.expanduser(self.h2_xr_scripts_dir)) if self.h2_xr_scripts_dir else None,
            Path(os.path.expanduser(self.workspace_root)) / 'H2_joint_control' / 'H2_joint_cartesian' / 'scripts',
            Path('/home/unitree/H2_joint_cartesian/scripts'),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)
        from h2_xr_official_ik_demo import H2CompatibleIK, current_ee_poses

        self._xr_ik = H2CompatibleIK()
        self._xr_current_ee_poses = current_ee_poses
        self.get_logger().info('xr_pinocchio FK initialized from %s' % (self.h2_xr_scripts_dir or '<auto>'))

    def _resolve_handeye_input(self, path):
        resolved = Path(os.path.expanduser(path))
        if resolved.suffix == '.json':
            npy_dir = resolved.with_name(resolved.stem + '_npy')
            if npy_dir.is_dir():
                return npy_dir
        return resolved

    def _load_matrix_from_candidates(self, root, candidates):
        root = self._resolve_handeye_input(root)
        if root.is_file():
            try:
                transform = np.load(str(root)).astype(np.float64)
            except Exception as exc:
                self.get_logger().warn('Failed to load transform npy %s: %s' % (root, exc))
                return None
            if transform.shape != (4, 4):
                self.get_logger().warn('transform must be 4x4, got shape=%s from %s' % (transform.shape, root))
                return None
            return transform, str(root)

        if root.is_dir():
            for name in candidates:
                candidate = root / name
                if candidate.is_file():
                    try:
                        transform = np.load(str(candidate)).astype(np.float64)
                    except Exception as exc:
                        self.get_logger().warn('Failed to load transform npy %s: %s' % (candidate, exc))
                        return None
                    if transform.shape != (4, 4):
                        self.get_logger().warn(
                            'transform must be 4x4, got shape=%s from %s' % (transform.shape, candidate)
                        )
                        return None
                    return transform, str(candidate)

        self.get_logger().warn('No supported transform found at: %s' % root)
        return None

    def _load_handeye_transform(self, path):
        if not path:
            return None

        loaded = self._load_matrix_from_candidates(
            path,
            ['T_cam2base.npy', 'T_cam2world.npy', 'T_camera2base.npy', 'T_camera2world.npy'],
        )
        if loaded is None:
            return None
        transform, source = loaded

        self.get_logger().info('Loaded hand-eye camera-to-target transform: %s' % source)
        return transform

    def _load_eye_in_hand_transform(self, path):
        loaded = self._load_matrix_from_candidates(
            path,
            ['T_cam2hand.npy', 'T_cam2ee.npy', 'T_camera2hand.npy', 'T_camera2ee.npy'],
        )
        if loaded is None:
            return None
        transform, source = loaded
        self.get_logger().info('Loaded eye-in-hand camera-to-hand transform: %s' % source)
        return transform

    def _output_frame_name(self):
        if self.T_cam2hand is not None:
            return self.handeye_target_frame or self.base_link or 'base_link'
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

    def _robot_status_callback(self, msg):
        self.latest_robot_status = msg

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
        joints = self._current_joint_values()
        if joints:
            self._publish_current_joint_state(joints, objects_msg.header)
        self._publish_objects_ik_json(out_msg, objects_msg.header, joints)
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

        if self.T_cam2hand is not None:
            transformed = self._transform_eye_in_hand(point_camera)
            if transformed is None:
                message = 'camera_frame; eye_in_hand_unavailable: %s' % (self._fk_error or 'missing joints/FK')
            else:
                point_target = transformed
                target_frame = self.handeye_target_frame or self.base_link or 'base_link'
                message = 'eye_in_hand_fk'
        elif self.handeye_transform is not None:
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
        if self.T_cam2hand is not None:
            msg.target_frame = self._output_frame_name()
        elif self.handeye_transform is not None:
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

    def _transform_eye_in_hand(self, point_camera):
        if self.T_cam2hand is None:
            return None
        joints = self._current_joint_values()
        if not joints:
            self._fk_error = 'missing joint values'
            return None
        try:
            T_base_hand = self._compute_base_to_hand(joints)
        except Exception as exc:
            self._fk_error = 'FK failed: %s' % exc
            self.get_logger().warn(self._fk_error)
            return None
        point_h = np.ones(4, dtype=np.float64)
        point_h[:3] = np.asarray(point_camera, dtype=np.float64).reshape(3)
        point_hand = self.T_cam2hand @ point_h
        point_target = T_base_hand @ point_hand
        return point_target[:3]

    def _current_joint_values(self):
        if self.joint_json_path:
            try:
                import json
                with open(os.path.expanduser(self.joint_json_path), 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                if isinstance(payload, dict) and 'joint_values' in payload:
                    payload = payload['joint_values']
                if isinstance(payload, dict) and 'joints' in payload:
                    payload = payload['joints']
                return {str(k): float(v) for k, v in payload.items()}
            except Exception as exc:
                self._fk_error = 'failed to load joint_json_path: %s' % exc
                return {}
        if self._joint_reader is None:
            return {}
        return self._joint_reader.joint_positions_by_name()

    def _compute_base_to_hand(self, joint_values):
        backend = self.fk_backend_requested or 'auto'
        if backend in ('auto', 'xr_pinocchio'):
            try:
                return self._compute_base_to_hand_xr(joint_values)
            except Exception as exc:
                self._fk_error = 'xr_pinocchio failed: %s' % exc
                if backend == 'xr_pinocchio':
                    raise
        if backend in ('auto', 'urdf'):
            return self._compute_base_to_hand_urdf(joint_values)
        raise RuntimeError('Unknown FK backend: %s' % backend)

    def _compute_base_to_hand_xr(self, joint_values):
        self._ensure_xr_fk()
        q = np.asarray(
            [float(joint_values.get(name, 0.0)) for name in H2_XR_DUAL_ARM_JOINTS],
            dtype=np.float64,
        )
        left_pose, right_pose = self._xr_current_ee_poses(self._xr_ik, q)
        pose = left_pose if self.hand_link.startswith('left_') else right_pose
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(pose.rotation, dtype=np.float64)
        transform[:3, 3] = np.asarray(pose.translation, dtype=np.float64).reshape(3)
        self.fk_backend_active = 'xr_pinocchio'
        self._fk_error = ''
        return transform

    def _compute_base_to_hand_urdf(self, joint_values):
        if self._fk_model is None:
            raise RuntimeError('URDF FK model is not initialized')
        targets = [self.hand_link]
        poses = self._fk_model.compute_link_poses(
            joint_values=self._lock_joint_values(joint_values),
            targets=targets,
            base_link=self.base_link or self._default_base_link(),
            base_pose=np.eye(4, dtype=np.float64),
            clamp_to_limits=False,
        )
        transform = np.asarray(poses[self.hand_link].matrix, dtype=np.float64)
        if self.robot_model == 'h2':
            offset = np.asarray(self.h2_ee_offset_xyz, dtype=np.float64).reshape(3)
            transform = transform.copy()
            transform[:3, 3] = transform[:3, 3] + transform[:3, :3] @ offset
        self.fk_backend_active = 'urdf'
        self._fk_error = ''
        return transform

    def _lock_joint_values(self, joint_values):
        out = dict(joint_values)
        if self.lock_waist:
            for name in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'):
                out[name] = 0.0
        return out

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

    def _publish_current_joint_state(self, joints, header):
        if not self.publish_current_joint_state:
            return
        msg = JointState()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.base_link or self._default_base_link()
        names = sorted(joints.keys())
        msg.name = names
        msg.position = [float(joints[name]) for name in names]
        self.current_joint_state_pub.publish(msg)

    def _current_joint_timer_callback(self):
        if not self.publish_current_joint_state:
            return
        joints = self._current_joint_values()
        if not joints:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_link or self._default_base_link()
        names = sorted(joints.keys())
        msg.name = names
        msg.position = [float(joints[name]) for name in names]
        self.current_joint_state_pub.publish(msg)

    def _publish_objects_ik_json(self, objects_3d_msg, header, joints):
        if not self.publish_objects_ik_json:
            return

        payload = {
            'header': {
                'stamp': {
                    'sec': int(header.stamp.sec),
                    'nanosec': int(header.stamp.nanosec),
                },
                'frame_id': str(header.frame_id),
            },
            'base_link': self.base_link or self._default_base_link(),
            'target_link': self.ik_target_link or self.hand_link or self._default_hand_link(),
            'current_joint_values_rad': {str(k): float(v) for k, v in (joints or {}).items()},
            'robot_status': self._robot_status_payload(self.latest_robot_status),
            'objects': [],
            'message': 'ok',
        }

        if not joints:
            payload['message'] = 'missing current joint values'
            self._publish_json_string(self.objects_ik_pub, payload)
            return
        if self._ik_solver is None:
            payload['message'] = 'IK solver unavailable: %s' % (self._ik_error or 'not initialized')
            self._publish_json_string(self.objects_ik_pub, payload)
            return

        solved = 0
        for index, object_3d in enumerate(objects_3d_msg.objects):
            item = {
                'object_id': self._object_id(index, object_3d.detection.class_name),
                'class_name': str(object_3d.detection.class_name),
                'class_id': int(object_3d.detection.class_id),
                'confidence': float(object_3d.detection.confidence),
                'bbox_xyxy': [
                    int(object_3d.detection.xmin),
                    int(object_3d.detection.ymin),
                    int(object_3d.detection.xmax),
                    int(object_3d.detection.ymax),
                ],
                'center_px': [float(object_3d.detection.cx), float(object_3d.detection.cy)],
                'valid_3d': bool(object_3d.valid),
                'target_frame': str(object_3d.target_frame),
                'point_target': self._point_payload(object_3d.point_target),
                'ik': None,
            }
            if not object_3d.valid:
                item['ik'] = {'success': False, 'message': object_3d.message or 'invalid 3D object'}
                payload['objects'].append(item)
                continue
            if solved >= max(0, self.max_ik_objects):
                item['ik'] = {'success': False, 'message': 'skipped: max_ik_objects reached'}
                payload['objects'].append(item)
                continue
            try:
                solution = self._solve_object_ik(object_3d, joints)
                item['ik'] = {
                    'success': bool(solution.success),
                    'message': str(solution.message),
                    'iterations': int(solution.iterations),
                    'position_error_m': float(solution.final_position_error_norm),
                    'orientation_error_rad': float(solution.final_orientation_error_norm),
                    'active_joints': [str(name) for name in solution.active_joints],
                    'joint_values_rad': {
                        str(name): float(solution.joint_values[name])
                        for name in solution.active_joints
                    },
                }
                solved += 1
            except Exception as exc:
                item['ik'] = {'success': False, 'message': 'IK failed: %s' % exc}
            payload['objects'].append(item)

        self._publish_json_string(self.objects_ik_pub, payload)

    def _publish_json_string(self, publisher, payload):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(msg)

    def _robot_status_payload(self, msg):
        if msg is None:
            return {
                'available': False,
                'stage_id': 0,
                'stage_name': '',
                'current_action': '',
                'motion_active': False,
                'progress': 0.0,
                'has_error': False,
                'error_code': '',
                'error_message': '',
                'emergency_stop': False,
                'target_reachable': False,
                'reachability_message': 'no /robot/inspection_status received',
                'target_id': '',
            }
        return {
            'available': True,
            'header': {
                'stamp': {
                    'sec': int(msg.header.stamp.sec),
                    'nanosec': int(msg.header.stamp.nanosec),
                },
                'frame_id': str(msg.header.frame_id),
            },
            'stage_id': int(msg.stage_id),
            'stage_name': str(msg.stage_name),
            'current_action': str(msg.current_action),
            'motion_active': bool(msg.motion_active),
            'progress': float(msg.progress),
            'has_error': bool(msg.has_error),
            'error_code': str(msg.error_code),
            'error_message': str(msg.error_message),
            'emergency_stop': bool(msg.emergency_stop),
            'target_reachable': bool(msg.target_reachable),
            'reachability_message': str(msg.reachability_message),
            'target_id': str(msg.target_id),
        }

    def _solve_object_ik(self, object_3d, joints):
        target_pose = self._ik_target_pose_matrix(object_3d.point_target)
        return self._ik_solver.solve(
            target_link=self.ik_target_link or self.hand_link or self._default_hand_link(),
            target_pose=target_pose,
            initial_joint_values=self._lock_joint_values(joints),
            active_joints=self.ik_active_joints or None,
            base_link=self.base_link or self._default_base_link(),
            base_pose=np.eye(4, dtype=np.float64).tolist(),
            max_iterations=self.ik_max_iterations,
            tolerance_position=self.ik_tolerance_position,
            tolerance_orientation=self.ik_tolerance_orientation,
            damping=self.ik_damping,
            step_scale=self.ik_step_scale,
            position_weight=self.ik_position_weight,
            orientation_weight=self.ik_orientation_weight,
            clamp_to_limits=True,
        )

    def _object_id(self, index, class_name):
        safe = ''.join(ch if ch.isalnum() else '_' for ch in str(class_name).lower()).strip('_')
        return '%02d_%s' % (int(index), safe or 'object')

    def _point_payload(self, point):
        return {'x': float(point.x), 'y': float(point.y), 'z': float(point.z)}

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

        self._publish_target_joint_state(object_3d, header)

    def _publish_target_joint_state(self, object_3d, header):
        if not self.publish_target_joint_state:
            return
        if self._ik_solver is None:
            self._warn_ik('IK solver is unavailable: %s' % (self._ik_error or 'not initialized'))
            return

        joints = self._current_joint_values()
        if not joints:
            self._warn_ik('IK skipped: missing current joint values')
            return

        try:
            target_pose = self._ik_target_pose_matrix(object_3d.point_target)
            solution = self._ik_solver.solve(
                target_link=self.ik_target_link or self.hand_link or self._default_hand_link(),
                target_pose=target_pose,
                initial_joint_values=self._lock_joint_values(joints),
                active_joints=self.ik_active_joints or None,
                base_link=self.base_link or self._default_base_link(),
                base_pose=np.eye(4, dtype=np.float64).tolist(),
                max_iterations=self.ik_max_iterations,
                tolerance_position=self.ik_tolerance_position,
                tolerance_orientation=self.ik_tolerance_orientation,
                damping=self.ik_damping,
                step_scale=self.ik_step_scale,
                position_weight=self.ik_position_weight,
                orientation_weight=self.ik_orientation_weight,
                clamp_to_limits=True,
            )
        except Exception as exc:
            self._warn_ik('IK failed: %s' % exc)
            return

        if not solution.success:
            self._warn_ik(
                'IK did not converge: %s, pos_err=%.4f, rot_err=%.4f'
                % (solution.message, solution.final_position_error_norm, solution.final_orientation_error_norm)
            )
            if not self.publish_failed_ik_solution:
                return

        joint_msg = JointState()
        joint_msg.header = header
        joint_msg.name = list(solution.active_joints)
        joint_msg.position = [float(solution.joint_values[name]) for name in solution.active_joints]
        self.target_joint_state_pub.publish(joint_msg)

    def _ik_target_pose_matrix(self, point_msg):
        qx, qy, qz, qw = self.pose_orientation_xyzw
        rotation = self._quat_to_matrix([qx, qy, qz, qw])
        position = np.array([point_msg.x, point_msg.y, point_msg.z], dtype=np.float64)
        position = position + np.asarray(self.ik_target_position_offset_xyz, dtype=np.float64).reshape(3)
        ee_offset = np.asarray(self.ik_end_effector_offset_xyz, dtype=np.float64).reshape(3)
        link_position = position - rotation @ ee_offset

        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = rotation
        target[:3, 3] = link_position
        return target.tolist()

    def _quat_to_matrix(self, quat_xyzw):
        x, y, z, w = [float(v) for v in quat_xyzw]
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _warn_ik(self, message):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self._last_ik_warning_sec >= 2.0:
            self.get_logger().warn(message)
            self._last_ik_warning_sec = now_sec

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
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
