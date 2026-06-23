# yolo_trt_ros2

这是一个面向 ROS2 Foxy 的轻量级 YOLO 检测 ROS 包，目标是在 Unitree G1 的 NVIDIA Jetson Orin NX 上运行，用于电柜操作任务中的 2D 目标检测。

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
