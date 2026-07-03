# Unitree H2 部署与基准测试指令集

这份指令面向 H2 PC2 / Jetson / Docker 上部署 `Foxy_ROS` 电柜巡检感知链路：

```text
RGB/Depth/CameraInfo
  -> yolo_detector_node
  -> /detector/objects
  -> coordinate_projector_node
  -> /detector/objects_3d, /detector/target_point, /detector/target_pose
```

H2 侧已有约定：

- DDS 网卡通常是 `eth0`，必要时改成实际网卡名。
- H2 机器人基座/身体坐标建议统一命名为 `pelvis` 或当前控制栈使用的 `base_link`。
- 右腕虚拟末端可参考 `h2_handeye` 的 `R_ee = right_wrist_yaw_link + [0.05, 0, 0] m`。
- 如果是 `eye-to-hand` 相机，优先使用 `T_cam2base.npy` 直接输出世界/基座坐标。
- 如果是 `eye-in-hand` 相机，单独 `T_cam2hand.npy` 还不够，需要实时 FK/TF 提供 `T_base_hand`。

## 0. 变量约定

在 Windows / WSL / H2 上把下面变量替换成实际值：

```text
H2_HOST=unitree@<H2-PC2-IP>
H2_WS=/home/unitree/MscapeTech
CONTAINER=wsdd_test
ROS_WS=/foxy_ros_custom
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

如果你的 H2 ROS/DDS 环境使用 CycloneDDS：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## 1. 本机传输到 H2

Windows PowerShell：

```powershell
scp -r E:\MscapeTech\Foxy_ROS unitree@<H2-PC2-IP>:~/tmp/
scp E:\MscapeTech\handle_recognition\minimal_test\yoloe-11s-seg.pt unitree@<H2-PC2-IP>:~/tmp/
```

WSL：

```bash
scp -r /mnt/e/MscapeTech/Foxy_ROS unitree@<H2-PC2-IP>:~/tmp/
scp /mnt/e/MscapeTech/handle_recognition/minimal_test/yoloe-11s-seg.pt unitree@<H2-PC2-IP>:~/tmp/
```

H2 宿主机上整理目录：

```bash
ssh unitree@<H2-PC2-IP>
mkdir -p ~/MscapeTech
rm -rf ~/MscapeTech/Foxy_ROS
mv ~/tmp/Foxy_ROS ~/MscapeTech/Foxy_ROS
mkdir -p ~/MscapeTech/models
mv ~/tmp/yoloe-11s-seg.pt ~/MscapeTech/models/
```

如果使用 Docker，把工作区拷进容器：

```bash
sudo docker exec ${CONTAINER} mkdir -p ${ROS_WS}
sudo docker cp ~/MscapeTech/Foxy_ROS/. ${CONTAINER}:${ROS_WS}/
sudo docker exec ${CONTAINER} mkdir -p ${ROS_WS}/models
sudo docker cp ~/MscapeTech/models/yoloe-11s-seg.pt ${CONTAINER}:${ROS_WS}/models/
```

## 2. 容器内环境检查

进入容器：

```bash
sudo docker exec -it ${CONTAINER} /bin/bash
```

避免 conda 污染 Foxy：

```bash
conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
source /opt/ros/foxy/setup.bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

安装基础依赖：

```bash
apt update
apt install -y \
  python3-opencv \
  python3-numpy \
  python3-colcon-common-extensions \
  ros-foxy-cv-bridge \
  ros-foxy-image-tools \
  ros-foxy-rqt-image-view \
  ros-foxy-rmw-cyclonedds-cpp
```

检查 Python/ROS：

```bash
which python3
python3 --version
python3 -c "import cv2; import numpy; import rclpy; import cv_bridge; print('ros python ok')"
```

如果要跑 `backend: yoloe`：

```bash
python3 -c "import ultralytics; print('ultralytics ok')"
```

如果没有 `ultralytics`，先用 `backend: mock` 完成 ROS 链路基准；YOLOE 依赖建议单独在 H2 的 Python 环境里安装和锁版本。

## 3. 编译 Foxy_ROS

```bash
cd /foxy_ros_custom
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

确认可执行入口：

```bash
ros2 pkg executables yolo_trt_ros2
ros2 interface show detector_msgs/msg/Object3D
```

## 4. 配置 H2 感知参数

编辑：

```bash
vi /foxy_ros_custom/src/yolo_trt_ros2/config/inspection_perception.yaml
```

推荐先用 mock：

```yaml
yolo_detector:
  ros__parameters:
    backend: mock
    publish_debug_image: true
```

切 YOLOE：

```yaml
yolo_detector:
  ros__parameters:
    backend: yoloe
    model_path: /foxy_ros_custom/models/yoloe-11s-seg.pt
    prompts: 'red button,green button,black knob,selector switch,rotary switch,toggle switch,control panel switch,cabinet door handle,indicator light,pilot light'
    conf_thres: 0.08
    iou_thres: 0.45
    imgsz: 640
```

如果相机是 `eye-to-hand`，配置 `T_cam2base.npy`：

```yaml
coordinate_projector:
  ros__parameters:
    handeye_npy_path: /foxy_ros_custom/outputs/eye_to_hand_xxx_npy/T_cam2base.npy
    handeye_target_frame: pelvis
```

如果是 `eye-in-hand`，当前 ROS 感知节点还需要实时 FK/TF 才能输出世界坐标。过渡方案：

- 先输出相机坐标：`handeye_npy_path: ''`，`target_frame: ''`。
- 或由 H2 FK/TF 节点发布 `pelvis -> camera_color_optical_frame`，再设置 `target_frame: pelvis`。
- `T_cam2hand.npy` 单独只能得到手腕/末端坐标，不能独立得到 `pelvis` 坐标。

## 5. 启动真实相机

如果使用 RealSense ROS2 wrapper，目标 topic 应对齐为：

```text
/camera/color/image_raw
/camera/aligned_depth_to_color/image_raw
/camera/color/camera_info
```

检查：

```bash
ros2 topic list
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/color/camera_info
```

没有真实相机时，可先发布测试 RGB 图像做 mock 基准：

```bash
ros2 run image_tools cam2image --ros-args -r image:=/camera/color/image_raw
```

如果 `cam2image` 试图打开 `/dev/video0` 且失败，说明容器没有挂载相机设备；这不影响后面的纯 ROS mock 测试，可用 README 中的 `/tmp/pub_test_image.py`。

## 6. 启动感知链路

终端 A：

```bash
cd /foxy_ros_custom
source /opt/ros/foxy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch yolo_trt_ros2 inspection_perception.launch.py
```

终端 B 查看输出：

```bash
source /opt/ros/foxy/setup.bash
source /foxy_ros_custom/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic echo /detector/objects
```

3D 坐标：

```bash
ros2 topic echo /detector/objects_3d
ros2 topic echo /detector/target_point
ros2 topic echo /detector/target_pose
```

调试图：

```bash
rqt_image_view /detector/debug_image
```

## 7. 基准测试 1：ROS 链路与 topic 频率

启动 benchmark 节点：

```bash
source /opt/ros/foxy/setup.bash
source /foxy_ros_custom/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 run yolo_trt_ros2 benchmark_topics_node --ros-args -p period_sec:=5.0
```

同时也可以用原生命令：

```bash
ros2 topic hz /camera/color/image_raw
ros2 topic hz /detector/objects
ros2 topic hz /detector/debug_image
ros2 topic hz /detector/objects_3d
ros2 topic hz /detector/target_point
```

记录模板：

```text
camera/color hz:
detector/objects hz:
detector/debug_image hz:
detector/objects_3d hz:
target_point hz:
header age avg/min/max:
```

判定建议：

- mock 后端：`/detector/objects` 应接近输入图像频率。
- YOLOE 后端：以实际模型速度为准，先记录 `imgsz=640`，再尝试降低到 `512` 做对比。
- `/detector/objects_3d` 低于 `/detector/objects` 时，优先检查深度图和 `CameraInfo` 是否同步、深度是否有效。

## 8. 基准测试 2：端到端坐标输出

启动完整链路后，查看一个目标的字段：

```bash
timeout 5 ros2 topic echo /detector/objects_3d
timeout 5 ros2 topic echo /detector/target_point
```

检查点：

```text
valid: true
depth_m: 合理，通常 0.2m 到 5.0m
source_frame: camera_color_optical_frame 或相机实际 frame
target_frame: pelvis/base_link/handeye_target_frame
point_camera: 单位 m
point_target: 单位 m
message: camera_frame / handeye_npy / transformed
```

如果启用了 `handeye_npy_path`，`message` 应为：

```text
handeye_npy
```

## 9. 基准测试 3：H2 FK 与手眼坐标一致性

只读 FK 对比，不发控制命令：

```bash
conda activate h1_arm
cd ~/MscapeTech/h2_handeye
python3 compare_h2_fk.py --iface eth0 --period 2.0 --count 5
```

如果要匹配 H2 键盘笛卡尔控制脚本的腰部锁定逻辑：

```bash
python3 compare_h2_fk.py --iface eth0 --period 2.0 --count 5 --lock-waist
```

记录模板：

```text
ours_xyz:
hw_xyz:
delta_mm:
norm_mm:
rot_deg:
lock_waist 是否显著改善:
```

建议阈值：

- FK 位置差：先以 `norm_mm < 5 mm` 作为工程基线。
- FK 姿态差：先以 `rot_deg < 1 deg` 作为工程基线。
- 若 `--lock-waist` 后误差明显变小，应统一感知链路和控制链路的腰部处理。

## 10. 基准测试 4：H2 运动前 dry-run

门面接近规划 dry-run：

```bash
cd ~/MscapeTech
python -m h2_pipeline.part1.run_live_realsense \
  --bbox 220 80 1050 690 \
  --iface eth0 \
  --stand-off 0.75
```

只检查 JSON，不移动机器人：

```text
success:
forward_m:
left_m:
yaw_rad:
stand_off_m:
```

确认方向和幅度正确、现场清空后，才允许执行：

```bash
python -m h2_pipeline.part1.run_live_realsense \
  --bbox 220 80 1050 690 \
  --iface eth0 \
  --stand-off 0.75 \
  --execute
```

## 11. 推荐跑测顺序

```text
1. colcon build 通过
2. mock + 假 RGB 图像，确认 /detector/objects
3. mock + 真实 RGB-D，确认 /detector/objects_3d
4. YOLOE + 真实 RGB-D，记录 hz 和 header age
5. 加 handeye_npy_path 或 TF，确认 target_frame 和 point_target
6. H2 compare_h2_fk.py，只读验证 FK 误差
7. h2_pipeline part1 dry-run，只看 JSON
8. 现场清空后再执行任何 H2 运动命令
```

## 12. 常见问题

### topic 看不到

```bash
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic list
```

必要时改 CycloneDDS：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### debug image 有，3D 没有

检查：

```bash
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/color/camera_info
ros2 topic echo /detector/objects
```

常见原因：

- 深度图没有对齐到 color。
- bbox 中心点处深度为空。
- `depth_scale` 不匹配，RealSense `16UC1` 通常是 `0.001` m/mm。

### YOLOE 启动失败

先切回 mock：

```yaml
backend: mock
```

确认 ROS 链路没问题后，再检查：

```bash
python3 -c "from ultralytics import YOLOE; print('YOLOE ok')"
ls -lh /foxy_ros_custom/models/yoloe-11s-seg.pt
```

### 手眼文件加载失败

`coordinate_projector_node` 当前直接支持：

```text
T_cam2base.npy
T_cam2world.npy
T_camera2base.npy
T_camera2world.npy
```

文件必须是 `4x4` 的 numpy 矩阵，且单位为米。
