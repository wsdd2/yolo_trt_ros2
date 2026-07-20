from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
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
    web_ui_arg = DeclareLaunchArgument(
        'webUI',
        default_value='false',
        description='Enable the in-process HTTP WebUI and preview generation.',
    )
    dex1_tip_arg = DeclareLaunchArgument(
        'dex1_tip_from_wrist_xyz',
        default_value='[0.14, 0.01, 0.012]',
        description='Dex1-1 fingertip/contact point offset from right_wrist_yaw_link, meters.',
    )
    blue_point_offset_arg = DeclareLaunchArgument(
        'blue_point_target_world_offset_xyz',
        default_value='[0.0, 0.001, -0.004]',
        description='World-frame offset applied only to red sticker push point targets, meters.',
    )
    handeye_mount_offset_arg = DeclareLaunchArgument(
        'handeye_mount_offset_from_wrist_xyz',
        default_value='[0.05, 0.0, 0.0]',
        description='Existing hand-eye calibration mount offset from right_wrist_yaw_link, meters.',
    )

    # Global ROS parameter overrides are harmless for the other in-process
    # nodes: only coordinate_projector declares these parameter names.
    ros_arguments = [
        '--ros-args',
        '--params-file',
        LaunchConfiguration('config_file'),
        '-p',
        ['dex1_tip_from_wrist_xyz:=', LaunchConfiguration('dex1_tip_from_wrist_xyz')],
        '-p',
        ['ik_end_effector_offset_xyz:=', LaunchConfiguration('handeye_mount_offset_from_wrist_xyz')],
        '-p',
        ['blue_point_target_world_offset_xyz:=', LaunchConfiguration('blue_point_target_world_offset_xyz')],
        '-p',
        ['handeye_mount_offset_from_wrist_xyz:=', LaunchConfiguration('handeye_mount_offset_from_wrist_xyz')],
    ]
    process_env = {
        'LD_LIBRARY_PATH': [
            '/opt/ros/humble/lib:',
            EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
        ],
    }
    process_prefix = ['/usr/bin/python3', '-m', 'yolo_trt_ros2.integrated_perception_node']

    integrated_node = ExecuteProcess(
        cmd=process_prefix + ros_arguments,
        additional_env=process_env,
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('webUI')),
    )

    integrated_node_with_web = ExecuteProcess(
        cmd=process_prefix + ['--webUI'] + ros_arguments,
        additional_env=process_env,
        output='screen',
        condition=IfCondition(LaunchConfiguration('webUI')),
    )

    return LaunchDescription([
        config_arg,
        web_ui_arg,
        dex1_tip_arg,
        blue_point_offset_arg,
        handeye_mount_offset_arg,
        integrated_node,
        integrated_node_with_web,
    ])
