from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_share = get_package_share_directory('yolo_trt_ros2')
    default_config = os.path.join(pkg_share, 'config', 'inspection_perception.yaml')

    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to detector and coordinate projector parameter YAML file.',
    )
    use_direct_camera_arg = DeclareLaunchArgument(
        'use_direct_camera',
        default_value='true',
        description='Start direct RealSense RGB-D publisher inside this launch.',
    )

    direct_camera_node = Node(
        package='yolo_trt_ros2',
        executable='direct_realsense_node',
        name='direct_realsense',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        condition=IfCondition(LaunchConfiguration('use_direct_camera')),
    )

    detector_node = Node(
        package='yolo_trt_ros2',
        executable='yolo_detector_node',
        name='yolo_detector',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    coordinate_node = Node(
        package='yolo_trt_ros2',
        executable='coordinate_projector_node',
        name='coordinate_projector',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    web_dashboard_node = Node(
        package='yolo_trt_ros2',
        executable='web_dashboard_node',
        name='web_dashboard',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    return LaunchDescription([
        config_arg,
        use_direct_camera_arg,
        direct_camera_node,
        detector_node,
        coordinate_node,
        web_dashboard_node,
    ])
