# yolo_trt_ros2

这是一个面向 ROS2 Foxy 的轻量级 YOLO 检测 ROS 包，目标是在 Unitree G1 的 NVIDIA Jetson Orin NX 上运行，用于机器人任务中的 2D 目标检测。

本包不依赖现成的 `yolo_ros`，也不强依赖 `ultralytics`、`torch` 或 amd64 环境。默认使用 `mock` 后端，用固定假检测框验证 ROS 图像订阅、检测结果发布和 debug image 发布流程。TensorRT 后端预留了接口，但不会在 `mock` 模式下导入 TensorRT。

## 目标平台

- 机器人：Unitree G1
- 计算平台：NVIDIA Jetson Orin NX
- 系统：Ubuntu 20.04
- ROS：ROS2 Foxy
- 架构：aarch64 / arm64
- JetPack / L4T：R35.3.1
- CUDA：11.4
- TensorRT：8.5.2

## 包结构

工作区结构如下：

```text
Foxy_ROS/
  src/
    detector_msgs/
      msg/
        Object2D.msg
        Object2DArray.msg
    yolo_trt_ros2/
      config/
        detector.yaml
      launch/
        yolo_detector.launch.py
      yolo_trt_ros2/
        yolo_detector_node.py
        backends/
          mock_backend.py
          tensorrt_backend.py
```

`detector_msgs` 提供简单的 2D 检测消息：

- `Object2D.msg`：单个目标，包括类别、置信度、bbox 和中心点。
- `Object2DArray.msg`：带 `std_msgs/Header` 的目标数组。

`yolo_trt_ros2` 提供检测节点：

- 订阅图像：`/camera/color/image_raw`
- 发布检测结果：`/detector/objects`
- 发布调试图像：`/detector/debug_image`

## 依赖安装

先确保已经安装并配置 ROS2 Foxy，然后安装常用依赖：

```bash
sudo apt update
sudo apt install -y \
  ros-foxy-rclpy \
  ros-foxy-sensor-msgs \
  ros-foxy-std-msgs \
  ros-foxy-cv-bridge \
  ros-foxy-rosidl-default-generators \
  ros-foxy-rosidl-default-runtime \
  python3-opencv \
  python3-numpy \
  python3-colcon-common-extensions
```

如果需要查看 debug image，可以安装：

```bash
sudo apt install -y ros-foxy-rqt-image-view ros-foxy-image-view
```

## 编译方法

进入工作区根目录：

```bash
cd ~/Foxy_ROS
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
```

如果你把 `src/detector_msgs` 和 `src/yolo_trt_ros2` 放到了已有的 ROS2 工作区，例如 `~/ros2_ws`：

```bash
cd ~/ros2_ws
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
```

## Source 方法

每次打开新终端后：

```bash
source /opt/ros/foxy/setup.bash
source install/setup.bash
```

也可以把 source 命令加入 `~/.bashrc`，但开发阶段建议手动 source，避免多个工作区互相影响。

## 启动方法

默认使用 `config/detector.yaml`，其中 `backend` 为 `mock`：

```bash
ros2 launch yolo_trt_ros2 yolo_detector.launch.py
```

使用自定义配置文件：

```bash
ros2 launch yolo_trt_ros2 yolo_detector.launch.py config_file:=/path/to/detector.yaml
```

## 参数说明

默认配置在 `config/detector.yaml`：

```yaml
yolo_detector:
  ros__parameters:
    image_topic: /camera/color/image_raw
    objects_topic: /detector/objects
    debug_image_topic: /detector/debug_image
    engine_path: ''
    class_names_path: ''
    input_width: 640
    input_height: 640
    conf_thres: 0.25
    iou_thres: 0.45
    publish_debug_image: true
    backend: mock
```

主要参数：

- `image_topic`：输入图像 topic。
- `objects_topic`：检测结果 topic。
- `debug_image_topic`：带检测框的调试图像 topic。
- `engine_path`：TensorRT engine 路径，只有 `backend=tensorrt` 时使用。
- `class_names_path`：类别名称文件路径，每行一个类别。
- `input_width` / `input_height`：模型输入尺寸，默认 640x640。
- `conf_thres`：置信度阈值。
- `iou_thres`：NMS IoU 阈值。
- `publish_debug_image`：是否发布 debug image。
- `backend`：后端类型，当前支持 `mock` 和预留的 `tensorrt`。

## 查看 Topic

启动节点后，查看 topic：

```bash
ros2 topic list
```

查看检测结果：

```bash
ros2 topic echo /detector/objects
```

查看发布频率：

```bash
ros2 topic hz /detector/objects
ros2 topic hz /detector/debug_image
```

## 查看 Debug Image

使用 `rqt_image_view`：

```bash
rqt_image_view /detector/debug_image
```

或使用 `image_view`：

```bash
ros2 run image_view image_view --ros-args -r image:=/detector/debug_image
```

## Mock 检测测试

`mock` 后端会在每帧输入图像上生成一个固定假检测框，用来验证 ROS 管线是否通畅。

测试步骤：

1. 启动相机或任意图像发布节点，确保有 `sensor_msgs/msg/Image` 发布到 `/camera/color/image_raw`。
2. 启动检测节点：

```bash
ros2 launch yolo_trt_ros2 yolo_detector.launch.py
```

3. 查看检测结果：

```bash
ros2 topic echo /detector/objects
```

正常情况下会看到类似结果：

```text
class_name: cabinet_door
class_id: 0
confidence: 0.9
xmin: ...
ymin: ...
xmax: ...
ymax: ...
cx: ...
cy: ...
```

4. 查看调试图像：

```bash
rqt_image_view /detector/debug_image
```

如果能看到绿色检测框，说明图像订阅、OpenCV 转换、检测结果发布和 debug image 发布流程都已经跑通。

## TensorRT 后端说明

`yolo_trt_ros2/backends/tensorrt_backend.py` 目前只预留接口，避免影响 `mock` 模式运行。TensorRT 相关 Python API 采用 lazy import，只有在配置 `backend: tensorrt` 时才会尝试导入。

后续实现 TensorRT 后端时，应保持统一接口：

```python
detections = backend.infer(bgr_image)
```

其中 `detections` 是 `list[dict]`，每个字典包含：

```text
class_name, class_id, confidence, xmin, ymin, xmax, ymax
```

后续可补充的 TensorRT 工作：

- 反序列化 `.engine` 文件。
- 分配 CUDA host/device buffer。
- 将 OpenCV BGR 图像预处理成模型输入。
- 执行 TensorRT inference。
- 解码 YOLO 输出。
- 执行 NMS。
- 输出统一 detection 字典格式。

## 注意事项

- 不使用 ROS2 Humble/Jazzy 才有的新 API。
- Python 代码兼容 ROS2 Foxy 常见的 Python 3.8 环境。
- 默认 `mock` 模式不需要 TensorRT、Torch 或 Ultralytics。
- 在 Jetson 上部署 TensorRT 时，需要保证 TensorRT Python bindings 与 JetPack/L4T 版本匹配。

## 远程宿主机与 Docker 测试记录

本节记录在 Unitree G1 Jetson Orin NX 远程宿主机和已有 Docker 容器中跑通 `mock` 检测管线的实际流程。

远程宿主机：

```text
unitree@192.168.1.96
```

进入已有容器：

```bash
sudo docker exec -it wsdd_test /bin/bash
```

### 从本机传输工作区

如果在 Windows PowerShell 中传输：

```powershell
scp -r E:\MscapeTech\Foxy_ROS unitree@192.168.1.96:~/tmp/
```

如果在 WSL 终端中传输，`E:\MscapeTech` 要写成 `/mnt/e/MscapeTech`：

```bash
scp -r /mnt/e/MscapeTech/Foxy_ROS unitree@192.168.1.96:~/tmp/
```

如果远程宿主机上只出现了 `~/tmp/src`，可以手动整理成 ROS2 workspace：

```bash
cd ~/tmp
mkdir -p Foxy_ros
mv src Foxy_ros/
ls Foxy_ros/src
```

应看到：

```text
detector_msgs  yolo_trt_ros2
```

注意 `~/tmp/Foxy_ros` 的真实路径是 `/home/unitree/tmp/Foxy_ros`，不是 `/tmp/Foxy_ros`。拷贝进 Docker 时使用宿主机真实路径：

```bash
sudo docker exec wsdd_test mkdir -p /foxy_ros_custom
sudo docker cp /home/unitree/tmp/Foxy_ros/. wsdd_test:/foxy_ros_custom/
```

### 容器内编译

进入容器：

```bash
sudo docker exec -it wsdd_test /bin/bash
```

编译：

```bash
cd /foxy_ros_custom
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

如果缺少 OpenCV、`cv_bridge` 或测试图像工具：

```bash
apt update
apt install -y \
  python3-opencv \
  python3-numpy \
  ros-foxy-cv-bridge \
  ros-foxy-image-tools
```

### 避免 conda 环境污染

ROS2 Foxy 通常使用系统 Python 3.8。Unitree 容器里可能存在 `unitree_sdk_py310` 或 `unitree_dds_py310` 这类 conda 环境，直接在 Python 3.10 conda 环境里运行 ROS2 Foxy 节点，可能导致 `cv2`、`rclpy`、`cv_bridge`、DDS 或动态库冲突。

启动 ROS2 节点前建议退出 conda 并清理关键变量：

```bash
conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
source /opt/ros/foxy/setup.bash
source /foxy_ros_custom/install/setup.bash
```

检查 Python 环境：

```bash
which python3
python3 --version
python3 -c "import cv2; import rclpy; import cv_bridge; print('ros python ok')"
```

理想情况下，`python3` 应来自 `/usr/bin/python3`。

### DDS 与 ROS2 daemon 排查

如果出现：

```text
bad_alloc caught: std::bad_alloc
Failed to confirm that the daemon started successfully
Killed
```

不一定是内存不足。先检查：

```bash
free -h
```

如果可用内存充足，优先怀疑 ROS2 daemon、RMW 或容器 DDS 环境。可以绕过 daemon，并固定 ROS domain 和 RMW：

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISABLE_DAEMON=1
```

部分 Foxy 版本的 `ros2 topic echo` 不支持 `--once`，直接使用：

```bash
ros2 topic echo /detector/objects
```

看到消息后按 `Ctrl+C` 停止。也可以用：

```bash
timeout 5 ros2 topic echo /detector/objects
```

如果 FastDDS 不稳定，可以尝试 CycloneDDS：

```bash
apt update
apt install -y ros-foxy-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

Docker 中 DDS 还可能受 `/dev/shm` 限制影响。如果后续新建测试容器，建议使用：

```bash
--net=host --ipc=host
```

或至少：

```bash
--net=host --shm-size=1g
```

### 无真实相机时的 mock 测试

`image_tools cam2image` 默认会尝试打开 `/dev/video0`。如果容器没有挂载真实相机，会报：

```text
Could not open video stream
```

这不是检测节点问题。测试 `mock` 管线时可以用一个临时 Python 节点发布纯色图像：

```bash
cat > /tmp/pub_test_image.py <<'PY'
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class PubImage(Node):
    def __init__(self):
        super().__init__('pub_test_image')
        self.pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.timer = self.create_timer(0.2, self.tick)

    def tick(self):
        w, h = 640, 480
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.height = h
        msg.width = w
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = bytes([40, 80, 160]) * (w * h)
        self.pub.publish(msg)

rclpy.init()
node = PubImage()
rclpy.spin(node)
PY
```

推荐开三个容器终端测试。

终端 1：启动检测节点：

```bash
cd /foxy_ros_custom
source /opt/ros/foxy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISABLE_DAEMON=1
ros2 launch yolo_trt_ros2 yolo_detector.launch.py
```

终端 2：发布测试图像：

```bash
source /opt/ros/foxy/setup.bash
source /foxy_ros_custom/install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISABLE_DAEMON=1
python3 /tmp/pub_test_image.py
```

终端 3：查看检测结果：

```bash
source /opt/ros/foxy/setup.bash
source /foxy_ros_custom/install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISABLE_DAEMON=1
ros2 topic echo /detector/objects
```

正常情况下会看到 `mock` 后端发布的假检测框：

```text
class_name: cabinet_door
class_id: 0
confidence: 0.9
xmin: ...
ymin: ...
xmax: ...
ymax: ...
cx: ...
cy: ...
```

如果后续接真实相机，需要在创建容器时挂载设备，例如 `--device=/dev/video0`，并先在宿主机确认 `/dev/video0` 存在。
