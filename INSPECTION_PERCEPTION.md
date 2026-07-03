# 电柜巡检 YOLO + ROS2 Foxy 通信

这个工作区现在包含两层感知通信：

- `yolo_detector_node`：订阅 RGB 图像，发布 2D 检测框。
- `coordinate_projector_node`：订阅 2D 检测、对齐深度图和 `CameraInfo`，发布相机系/目标系 3D 坐标。

## 主要 Topic

输入：

- `/camera/color/image_raw`：RGB 图像，`sensor_msgs/Image`
- `/camera/aligned_depth_to_color/image_raw`：对齐到 RGB 的深度图，`sensor_msgs/Image`
- `/camera/color/camera_info`：RGB 相机内参，`sensor_msgs/CameraInfo`

输出：

- `/detector/objects`：2D 检测结果，`detector_msgs/Object2DArray`
- `/detector/debug_image`：带框调试图，`sensor_msgs/Image`
- `/detector/objects_3d`：每个检测目标的 3D 坐标，`detector_msgs/Object3DArray`
- `/detector/target_point`：当前最佳目标点，`geometry_msgs/PointStamped`
- `/detector/target_pose`：当前最佳目标位姿，`geometry_msgs/PoseStamped`

## 启动

在 Jetson / Docker 的 Foxy 环境中：

```bash
cd /foxy_ros_custom
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch yolo_trt_ros2 inspection_perception.launch.py
```

默认配置文件是：

```text
src/yolo_trt_ros2/config/inspection_perception.yaml
```

如果权重不在默认路径 `/foxy_ros_custom/models/yoloe-11s-seg.pt`，改这个参数：

```yaml
yolo_detector:
  ros__parameters:
    model_path: /your/path/yoloe-11s-seg.pt
```

## 坐标系

默认 `target_frame: ''`，因此 `/detector/target_point` 和 `/detector/target_pose` 使用相机光学坐标系，也就是检测图像 header 里的 `frame_id`。

如果已经有手眼标定 TF，例如 `base_link <- camera_color_optical_frame`，可以设置：

```yaml
coordinate_projector:
  ros__parameters:
    target_frame: base_link
```

这样 `Object3D.point_camera` 保留相机系坐标，`Object3D.point_target` 和 `/detector/target_point` 输出 `base_link` 坐标。

## 快速检查

```bash
ros2 topic echo /detector/objects
ros2 topic echo /detector/objects_3d
ros2 topic echo /detector/target_point
ros2 topic hz /detector/debug_image
```

如果只想先验证 ROS 通信，把配置里的 `backend` 改为 `mock`，无需安装 `ultralytics` 或加载真实模型。
