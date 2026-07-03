#!/usr/bin/env python3
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from detector_msgs.msg import Object2DArray, Object3DArray


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inspection Perception Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101418;
      --panel: #171d22;
      --panel-2: #1f272e;
      --line: #33404a;
      --text: #e6edf3;
      --muted: #9ba8b3;
      --ok: #33d17a;
      --warn: #f6d365;
      --bad: #ff6b6b;
      --accent: #6cb6ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #0d1115;
    }
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(420px, 1.45fr) minmax(360px, 0.9fr);
      gap: 14px;
      padding: 14px;
      min-height: calc(100vh - 52px);
    }
    section {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .video-wrap {
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .section-head {
      height: 42px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 0 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--muted);
      font-size: 12px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--bad);
    }
    .dot.ok { background: var(--ok); }
    .dot.warn { background: var(--warn); }
    .video {
      width: 100%;
      height: 100%;
      min-height: 360px;
      object-fit: contain;
      background: #050708;
      display: block;
    }
    .side {
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .content {
      padding: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      text-align: left;
      padding: 7px 6px;
      border-bottom: 1px solid rgba(255,255,255,0.07);
      vertical-align: top;
      font-variant-numeric: tabular-nums;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    .kv {
      display: grid;
      grid-template-columns: 128px 1fr;
      gap: 8px 12px;
      font-variant-numeric: tabular-nums;
    }
    .key { color: var(--muted); }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      word-break: break-word;
    }
    .empty {
      color: var(--muted);
      padding: 12px 0;
    }
    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; }
      .video { min-height: 260px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Inspection Perception Dashboard</h1>
    <div class="status"><span id="dot" class="dot"></span><span id="status">connecting</span></div>
  </header>
  <main>
    <section class="video-wrap">
      <div class="section-head">
        <span>debug image stream</span>
        <span id="imageMeta" class="mono">--</span>
      </div>
      <img class="video" src="/stream.mjpg" alt="debug image stream">
    </section>
    <div class="side">
      <section>
        <div class="section-head"><span>2D detections</span><span id="objectMeta" class="mono">--</span></div>
        <div class="content">
          <table>
            <thead><tr><th>class</th><th>conf</th><th>bbox</th><th>center</th></tr></thead>
            <tbody id="objects2d"></tbody>
          </table>
        </div>
      </section>
      <section>
        <div class="section-head"><span>3D target</span><span id="targetMeta" class="mono">--</span></div>
        <div class="content">
          <div id="target3d" class="kv"></div>
        </div>
      </section>
      <section>
        <div class="section-head"><span>IK joint state</span><span id="jointMeta" class="mono">--</span></div>
        <div class="content">
          <table>
            <thead><tr><th>joint</th><th>position rad</th></tr></thead>
            <tbody id="joints"></tbody>
          </table>
        </div>
      </section>
    </div>
  </main>
  <script>
    const fmt = (v, digits = 3) => Number.isFinite(v) ? Number(v).toFixed(digits) : "--";
    const stamp = h => h && h.stamp ? `${h.stamp.sec}.${String(h.stamp.nanosec).padStart(9, "0")}` : "--";
    function setRows(tbody, rows, emptyCols) {
      tbody.innerHTML = "";
      if (!rows.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = emptyCols;
        td.className = "empty";
        td.textContent = "no data";
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
      }
      rows.forEach(row => tbody.appendChild(row));
    }
    function kv(el, pairs) {
      el.innerHTML = "";
      if (!pairs.length) {
        const div = document.createElement("div");
        div.className = "empty";
        div.textContent = "no data";
        el.appendChild(div);
        return;
      }
      pairs.forEach(([k, v]) => {
        const key = document.createElement("div");
        key.className = "key";
        key.textContent = k;
        const val = document.createElement("div");
        val.className = "mono";
        val.textContent = v;
        el.appendChild(key);
        el.appendChild(val);
      });
    }
    async function refresh() {
      try {
        const res = await fetch("/api/state", { cache: "no-store" });
        const data = await res.json();
        const age = data.server_time - data.last_update_time;
        const dot = document.getElementById("dot");
        dot.className = "dot " + (age < 1.5 ? "ok" : age < 5 ? "warn" : "");
        document.getElementById("status").textContent = `age ${fmt(age, 2)}s`;
        document.getElementById("imageMeta").textContent = `${data.image.width || "--"}x${data.image.height || "--"} q=${data.image.jpeg_quality}`;
        document.getElementById("objectMeta").textContent = `${data.objects2d.objects.length} objects`;
        const objectRows = data.objects2d.objects.map(o => {
          const tr = document.createElement("tr");
          [o.class_name, fmt(o.confidence, 2), `${o.xmin},${o.ymin},${o.xmax},${o.ymax}`, `${fmt(o.cx, 1)},${fmt(o.cy, 1)}`]
            .forEach(text => {
              const td = document.createElement("td");
              td.className = "mono";
              td.textContent = text;
              tr.appendChild(td);
            });
          return tr;
        });
        setRows(document.getElementById("objects2d"), objectRows, 4);
        const best3d = data.objects3d.objects.find(o => o.valid) || data.objects3d.objects[0];
        document.getElementById("targetMeta").textContent = best3d ? (best3d.target_frame || "--") : "--";
        if (best3d) {
          kv(document.getElementById("target3d"), [
            ["class", best3d.detection.class_name],
            ["confidence", fmt(best3d.detection.confidence, 3)],
            ["depth_m", fmt(best3d.depth_m, 3)],
            ["source_frame", best3d.source_frame || "--"],
            ["target_frame", best3d.target_frame || "--"],
            ["point_camera", `${fmt(best3d.point_camera.x)}, ${fmt(best3d.point_camera.y)}, ${fmt(best3d.point_camera.z)}`],
            ["point_target", `${fmt(best3d.point_target.x)}, ${fmt(best3d.point_target.y)}, ${fmt(best3d.point_target.z)}`],
            ["message", best3d.message || "--"],
            ["stamp", stamp(best3d.header)]
          ]);
        } else if (data.target_pose.header) {
          const p = data.target_pose.pose.position;
          kv(document.getElementById("target3d"), [
            ["target_frame", data.target_pose.header.frame_id || "--"],
            ["pose_xyz", `${fmt(p.x)}, ${fmt(p.y)}, ${fmt(p.z)}`],
            ["stamp", stamp(data.target_pose.header)]
          ]);
        } else {
          kv(document.getElementById("target3d"), []);
        }
        document.getElementById("jointMeta").textContent = `${data.target_joint_state.name.length} joints`;
        const jointRows = data.target_joint_state.name.map((name, i) => {
          const tr = document.createElement("tr");
          [name, fmt(data.target_joint_state.position[i], 5)].forEach(text => {
            const td = document.createElement("td");
            td.className = "mono";
            td.textContent = text;
            tr.appendChild(td);
          });
          return tr;
        });
        setRows(document.getElementById("joints"), jointRows, 2);
      } catch (err) {
        document.getElementById("dot").className = "dot";
        document.getElementById("status").textContent = "disconnected";
      }
    }
    setInterval(refresh, 300);
    refresh();
  </script>
</body>
</html>
"""


class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_condition = threading.Condition(self.lock)
        self.jpeg = None
        self.image_header = None
        self.image_width = 0
        self.image_height = 0
        self.jpeg_quality = 80
        self.last_update_time = 0.0
        self.objects2d = None
        self.objects3d = None
        self.target_point = None
        self.target_pose = None
        self.target_joint_state = None


class WebDashboardNode(Node):
    """Serve a tiny browser dashboard for detector video and perception state."""

    def __init__(self):
        super().__init__('web_dashboard')
        self._declare_parameters()

        self.host = str(self.get_parameter('web_host').value)
        self.port = int(self.get_parameter('web_port').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.frame_timeout_sec = float(self.get_parameter('frame_timeout_sec').value)
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.objects_topic = self.get_parameter('objects_topic').value
        self.objects_3d_topic = self.get_parameter('objects_3d_topic').value
        self.target_point_topic = self.get_parameter('target_point_topic').value
        self.target_pose_topic = self.get_parameter('target_pose_topic').value
        self.target_joint_state_topic = self.get_parameter('target_joint_state_topic').value

        self.bridge = CvBridge()
        self.state = DashboardState()
        self.state.jpeg_quality = max(1, min(100, self.jpeg_quality))

        self.create_subscription(Image, self.debug_image_topic, self._image_callback, 10)
        self.create_subscription(Object2DArray, self.objects_topic, self._objects_callback, 10)
        self.create_subscription(Object3DArray, self.objects_3d_topic, self._objects3d_callback, 10)
        self.create_subscription(PointStamped, self.target_point_topic, self._target_point_callback, 10)
        self.create_subscription(PoseStamped, self.target_pose_topic, self._target_pose_callback, 10)
        self.create_subscription(JointState, self.target_joint_state_topic, self._joint_state_callback, 10)

        handler_cls = self._make_handler()
        self.server = ThreadingHTTPServer((self.host, self.port), handler_cls)
        self.server.daemon_threads = True
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.get_logger().info(
            'Web dashboard started: http://%s:%d/ image=%s objects=%s objects_3d=%s'
            % (self.host if self.host != '0.0.0.0' else '<H2-IP>', self.port, self.debug_image_topic, self.objects_topic, self.objects_3d_topic)
        )

    def _declare_parameters(self):
        self.declare_parameter('web_host', '0.0.0.0')
        self.declare_parameter('web_port', 8080)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('frame_timeout_sec', 2.0)
        self.declare_parameter('debug_image_topic', '/detector/debug_image')
        self.declare_parameter('objects_topic', '/detector/objects')
        self.declare_parameter('objects_3d_topic', '/detector/objects_3d')
        self.declare_parameter('target_point_topic', '/detector/target_point')
        self.declare_parameter('target_pose_topic', '/detector/target_pose')
        self.declare_parameter('target_joint_state_topic', '/detector/target_joint_state')

    def _image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.state.jpeg_quality])
        except Exception as exc:
            self.get_logger().warn('Failed to encode dashboard image: %s' % exc)
            return
        if not ok:
            self.get_logger().warn('Failed to encode dashboard image as JPEG.')
            return

        with self.state.frame_condition:
            self.state.jpeg = encoded.tobytes()
            self.state.image_header = msg.header
            self.state.image_width = int(msg.width)
            self.state.image_height = int(msg.height)
            self.state.last_update_time = time.time()
            self.state.frame_condition.notify_all()

    def _objects_callback(self, msg):
        with self.state.lock:
            self.state.objects2d = msg
            self.state.last_update_time = time.time()

    def _objects3d_callback(self, msg):
        with self.state.lock:
            self.state.objects3d = msg
            self.state.last_update_time = time.time()

    def _target_point_callback(self, msg):
        with self.state.lock:
            self.state.target_point = msg
            self.state.last_update_time = time.time()

    def _target_pose_callback(self, msg):
        with self.state.lock:
            self.state.target_pose = msg
            self.state.last_update_time = time.time()

    def _joint_state_callback(self, msg):
        with self.state.lock:
            self.state.target_joint_state = msg
            self.state.last_update_time = time.time()

    def _make_handler(self):
        state = self.state
        frame_timeout_sec = self.frame_timeout_sec

        class DashboardHandler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self._send_bytes(INDEX_HTML.encode('utf-8'), 'text/html; charset=utf-8')
                    return
                if self.path.startswith('/api/state'):
                    self._send_json(self._snapshot())
                    return
                if self.path.startswith('/stream.mjpg'):
                    self._stream_mjpeg()
                    return
                self.send_error(404, 'not found')

            def log_message(self, fmt, *args):
                return

            def _send_bytes(self, payload, content_type):
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(payload)

            def _send_json(self, payload):
                data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                self._send_bytes(data, 'application/json; charset=utf-8')

            def _stream_mjpeg(self):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                last_frame = None
                while True:
                    with state.frame_condition:
                        state.frame_condition.wait(timeout=frame_timeout_sec)
                        frame = state.jpeg
                    if frame is None or frame == last_frame:
                        continue
                    last_frame = frame
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(('Content-Length: %d\r\n\r\n' % len(frame)).encode('ascii'))
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    except (BrokenPipeError, ConnectionResetError):
                        return

            def _snapshot(self):
                with state.lock:
                    return {
                        'server_time': time.time(),
                        'last_update_time': state.last_update_time,
                        'image': {
                            'header': header_to_dict(state.image_header),
                            'width': state.image_width,
                            'height': state.image_height,
                            'jpeg_quality': state.jpeg_quality,
                        },
                        'objects2d': object2d_array_to_dict(state.objects2d),
                        'objects3d': object3d_array_to_dict(state.objects3d),
                        'target_point': point_stamped_to_dict(state.target_point),
                        'target_pose': pose_stamped_to_dict(state.target_pose),
                        'target_joint_state': joint_state_to_dict(state.target_joint_state),
                    }

        return DashboardHandler

    def destroy_node(self):
        try:
            self.server.shutdown()
            self.server.server_close()
        finally:
            super().destroy_node()


def stamp_to_dict(stamp):
    if stamp is None:
        return {'sec': 0, 'nanosec': 0}
    return {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)}


def header_to_dict(header):
    if header is None:
        return None
    return {'stamp': stamp_to_dict(header.stamp), 'frame_id': str(header.frame_id)}


def point_to_dict(point):
    if point is None:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}
    return {'x': float(point.x), 'y': float(point.y), 'z': float(point.z)}


def quaternion_to_dict(quat):
    if quat is None:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
    return {'x': float(quat.x), 'y': float(quat.y), 'z': float(quat.z), 'w': float(quat.w)}


def object2d_to_dict(obj):
    return {
        'class_name': str(obj.class_name),
        'class_id': int(obj.class_id),
        'confidence': float(obj.confidence),
        'xmin': int(obj.xmin),
        'ymin': int(obj.ymin),
        'xmax': int(obj.xmax),
        'ymax': int(obj.ymax),
        'cx': float(obj.cx),
        'cy': float(obj.cy),
    }


def object2d_array_to_dict(msg):
    if msg is None:
        return {'header': None, 'objects': []}
    return {
        'header': header_to_dict(msg.header),
        'objects': [object2d_to_dict(obj) for obj in msg.objects],
    }


def object3d_to_dict(obj):
    return {
        'header': header_to_dict(obj.header),
        'detection': object2d_to_dict(obj.detection),
        'valid': bool(obj.valid),
        'cached': bool(obj.cached),
        'source_frame': str(obj.source_frame),
        'target_frame': str(obj.target_frame),
        'depth_m': float(obj.depth_m),
        'point_camera': point_to_dict(obj.point_camera),
        'point_target': point_to_dict(obj.point_target),
        'message': str(obj.message),
    }


def object3d_array_to_dict(msg):
    if msg is None:
        return {'header': None, 'objects': []}
    return {
        'header': header_to_dict(msg.header),
        'objects': [object3d_to_dict(obj) for obj in msg.objects],
    }


def point_stamped_to_dict(msg):
    if msg is None:
        return {'header': None, 'point': point_to_dict(None)}
    return {'header': header_to_dict(msg.header), 'point': point_to_dict(msg.point)}


def pose_stamped_to_dict(msg):
    if msg is None:
        return {
            'header': None,
            'pose': {
                'position': point_to_dict(None),
                'orientation': quaternion_to_dict(None),
            },
        }
    return {
        'header': header_to_dict(msg.header),
        'pose': {
            'position': point_to_dict(msg.pose.position),
            'orientation': quaternion_to_dict(msg.pose.orientation),
        },
    }


def joint_state_to_dict(msg):
    if msg is None:
        return {'header': None, 'name': [], 'position': [], 'velocity': [], 'effort': []}
    return {
        'header': header_to_dict(msg.header),
        'name': [str(name) for name in msg.name],
        'position': [float(value) for value in msg.position],
        'velocity': [float(value) for value in msg.velocity],
        'effort': [float(value) for value in msg.effort],
    }


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WebDashboardNode()
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
