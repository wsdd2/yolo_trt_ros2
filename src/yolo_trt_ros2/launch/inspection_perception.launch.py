from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
        detector_node,
        coordinate_node,
        web_dashboard_node,
    ])
