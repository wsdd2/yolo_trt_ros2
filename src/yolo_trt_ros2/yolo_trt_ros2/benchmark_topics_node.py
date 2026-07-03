#!/usr/bin/env python3
import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

from detector_msgs.msg import Object2DArray, Object3DArray


@dataclass
class TopicStats:
    name: str
    count: int = 0
    last_count: int = 0
    age_count: int = 0
    age_sum_ms: float = 0.0
    age_min_ms: float = math.inf
    age_max_ms: float = 0.0

    def push_age(self, age_ms):
        self.count += 1
        if age_ms is None:
            return
        self.age_count += 1
        self.age_sum_ms += age_ms
        self.age_min_ms = min(self.age_min_ms, age_ms)
        self.age_max_ms = max(self.age_max_ms, age_ms)

    def snapshot(self, period_s):
        delta = self.count - self.last_count
        self.last_count = self.count
        hz = float(delta) / max(float(period_s), 1e-6)
        if self.age_count > 0:
            avg_age = self.age_sum_ms / float(self.age_count)
            min_age = 0.0 if self.age_min_ms == math.inf else self.age_min_ms
            max_age = self.age_max_ms
            return hz, avg_age, min_age, max_age
        return hz, None, None, None


class BenchmarkTopicsNode(Node):
    """Print lightweight rate and header-age metrics for the perception topics."""

    def __init__(self):
        super().__init__('perception_benchmark')
        self.declare_parameter('period_sec', 5.0)
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('objects_topic', '/detector/objects')
        self.declare_parameter('debug_image_topic', '/detector/debug_image')
        self.declare_parameter('objects_3d_topic', '/detector/objects_3d')
        self.declare_parameter('target_point_topic', '/detector/target_point')
        self.declare_parameter('target_pose_topic', '/detector/target_pose')

        self.period_sec = float(self.get_parameter('period_sec').value)
        topics = [
            (self.get_parameter('image_topic').value, Image),
            (self.get_parameter('objects_topic').value, Object2DArray),
            (self.get_parameter('debug_image_topic').value, Image),
            (self.get_parameter('objects_3d_topic').value, Object3DArray),
            (self.get_parameter('target_point_topic').value, PointStamped),
            (self.get_parameter('target_pose_topic').value, PoseStamped),
        ]

        self.stats = {}
        self._bench_subscriptions = []
        for topic_name, msg_type in topics:
            if not topic_name:
                continue
            self.stats[topic_name] = TopicStats(name=topic_name)
            self._bench_subscriptions.append(
                self.create_subscription(
                    msg_type,
                    topic_name,
                    self._make_callback(topic_name),
                    10,
                )
            )

        self.timer = self.create_timer(self.period_sec, self._print_stats)
        self.get_logger().info('Perception benchmark started with period %.2fs' % self.period_sec)

    def _make_callback(self, topic_name):
        def _callback(msg):
            self.stats[topic_name].push_age(self._header_age_ms(msg))

        return _callback

    def _header_age_ms(self, msg):
        header = getattr(msg, 'header', None)
        if header is None:
            return None
        stamp = getattr(header, 'stamp', None)
        if stamp is None:
            return None
        stamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if stamp_s <= 0.0:
            return None
        now_msg = self.get_clock().now().to_msg()
        now_s = float(now_msg.sec) + float(now_msg.nanosec) * 1e-9
        return max(0.0, (now_s - stamp_s) * 1000.0)

    def _print_stats(self):
        lines = ['benchmark window=%.2fs' % self.period_sec]
        for topic_name in sorted(self.stats):
            hz, avg_age, min_age, max_age = self.stats[topic_name].snapshot(self.period_sec)
            if avg_age is None:
                lines.append('%s hz=%.2f age_ms=N/A count=%d' % (topic_name, hz, self.stats[topic_name].count))
            else:
                lines.append(
                    '%s hz=%.2f age_ms(avg/min/max)=%.1f/%.1f/%.1f count=%d'
                    % (topic_name, hz, avg_age, min_age, max_age, self.stats[topic_name].count)
                )
        self.get_logger().info('\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BenchmarkTopicsNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
