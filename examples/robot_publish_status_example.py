#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot-side example: publish inspection status back to perception.

Topic:
  /robot/inspection_status  detector_msgs/msg/RobotInspectionStatus

The perception side subscribes to this topic and mirrors it into:
  - /detector/objects_ik_json -> robot_status
  - Web dashboard /api/state -> robot_status

Run on H2:
  cd ~/MscapeTech/Foxy_ROS
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  export ROS_DOMAIN_ID=42
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  python3 examples/robot_publish_status_example.py --stage 2 --reachable true
"""

import argparse

import rclpy
from rclpy.node import Node

from detector_msgs.msg import RobotInspectionStatus


STAGE_NAMES = {
    0: "idle",
    1: "move_to_handle_front",
    2: "press_blue_point",
    3: "grasp_or_pull_handle",
    4: "door_opened",
    5: "recover_or_abort",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish robot inspection status.")
    parser.add_argument("--stage", type=int, default=0, help="0 idle, 1 approach, 2 press, 3 pull, 4 done, 5 recover")
    parser.add_argument("--stage-name", default="", help="Override stage name.")
    parser.add_argument("--action", default="", help="Current robot action.")
    parser.add_argument("--active", action="store_true", help="Robot motion is currently active.")
    parser.add_argument("--progress", type=float, default=0.0, help="Stage progress, usually 0.0 to 1.0.")
    parser.add_argument("--reachable", choices=("true", "false"), default="true", help="Whether current target is reachable.")
    parser.add_argument("--reachability-message", default="ok", help="Reachability detail.")
    parser.add_argument("--target-id", default="", help="Object id being acted on, e.g. 01_blue_push_point.")
    parser.add_argument("--error", action="store_true", help="Robot has an error.")
    parser.add_argument("--error-code", default="", help="Robot-side error code.")
    parser.add_argument("--error-message", default="", help="Robot-side error message.")
    parser.add_argument("--estop", action="store_true", help="Emergency stop is active.")
    parser.add_argument("--rate", type=float, default=5.0, help="Publish rate in Hz.")
    return parser.parse_args()


class RobotStatusPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("robot_status_publisher_example")
        self.args = args
        self.pub = self.create_publisher(RobotInspectionStatus, "/robot/inspection_status", 10)
        self.timer = self.create_timer(1.0 / max(0.1, float(args.rate)), self._publish_status)
        self.get_logger().info("Publishing: /robot/inspection_status")

    def _publish_status(self) -> None:
        msg = RobotInspectionStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "pelvis"

        msg.stage_id = int(self.args.stage)
        msg.stage_name = self.args.stage_name or STAGE_NAMES.get(msg.stage_id, "unknown")
        msg.current_action = self.args.action or msg.stage_name
        msg.motion_active = bool(self.args.active)
        msg.progress = float(max(0.0, min(1.0, self.args.progress)))

        msg.has_error = bool(self.args.error)
        msg.error_code = str(self.args.error_code)
        msg.error_message = str(self.args.error_message)
        msg.emergency_stop = bool(self.args.estop)

        msg.target_reachable = self.args.reachable == "true"
        msg.reachability_message = str(self.args.reachability_message)
        msg.target_id = str(self.args.target_id)

        self.pub.publish(msg)


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = RobotStatusPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
