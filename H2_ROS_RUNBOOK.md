# H2 ROS 感知启动与排障速查

本文记录 Unitree H2 上 `Foxy_ROS`（实际为 Humble）当前遇到过的启动分支、常见报错和对应命令。目标是让现场优先跑通：

```text
RealSense RGB-D -> YOLOE 2D -> 3D 世界坐标 -> IK 关节目标 -> Web/ROS topics
```

默认 H2：

```text
H2 IP: 192.168.25.189
ROS: Humble
Workspace: /home/unitree/MscapeTech/Foxy_ROS
Unitree SDK: /home/unitree/MscapeTech/unitree_sdk2_python
Hand-eye: /home/unitree/MscapeTech/Hand_Eye_Calib/outputs/eye_in_hand_20260630_150210.json
Lowstate: eth0, domain 0, rt/lf/lowstate
Web: http://192.168.25.189:8080/ 或 8081
```

## 1. 从本地同步最新版

在 WSL 执行，不带 `--delete`，避免删除 H2 现场私有文件：

```bash
rsync -av --progress \
  --exclude build \
  --exclude install \
  --exclude log \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  /mnt/e/MscapeTech/Foxy_ROS/ \
  unitree@192.168.25.189:/home/unitree/MscapeTech/Foxy_ROS/
```

如果只同步直连脚本：

```bash
rsync -av --progress \
  /mnt/e/MscapeTech/handle_recognition/minimal_test/run_h2_handle_pose_direct.py \
  unitree@192.168.25.189:/home/unitree/MscapeTech/handle_recognition/minimal_test/run_h2_handle_pose_direct.py
```

## 2. ROS 编译

Humble 编译不要在 conda 里做：

```bash
cd ~/MscapeTech/Foxy_ROS

conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/humble/setup.bash

rm -rf build/yolo_trt_ros2 install/yolo_trt_ros2

colcon build --packages-select detector_msgs yolo_trt_ros2

source install/setup.bash

ros2 pkg executables yolo_trt_ros2
```

应该至少看到：

```text
yolo_trt_ros2 direct_realsense_node
yolo_trt_ros2 yolo_detector_node
yolo_trt_ros2 coordinate_projector_node
yolo_trt_ros2 web_dashboard_node
```

验证 Python 包 metadata：

```bash
python3 -c "import importlib.metadata as m; print(m.distribution('yolo-trt-ros2'))"
python3 -c "import yolo_trt_ros2.coordinate_projector_node; print('coordinate import ok')"
```

## 3. 推荐的一体化 ROS 启动

启动前必须先 source，再把 Unitree SDK 前置到 `PYTHONPATH`，不要覆盖整个 `PYTHONPATH`：

```bash
cd ~/MscapeTech/Foxy_ROS

conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/humble/setup.bash
source install/setup.bash

export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml \
  use_direct_camera:=true
```

启动成功后：

```bash
ros2 node list
```

应看到：

```text
/direct_realsense
/yolo_detector
/coordinate_projector
/web_dashboard
```

核心 topic：

```bash
ros2 topic echo --once /camera/color/camera_info
ros2 topic echo --once /detector/objects
ros2 topic echo --once /detector/objects_3d
ros2 topic echo --once /detector/current_joint_state
ros2 topic echo --once /detector/target_joint_state
ros2 topic echo --once /detector/objects_ik_json
```

网页：

```text
http://192.168.25.189:8080/
```

## 4. 端口 8080 被占用

如果报：

```text
OSError: [Errno 98] Address already in use
```

查占用：

```bash
sudo lsof -i :8080
```

或把配置改到 8081：

```bash
sed -i 's/web_port: 8080/web_port: 8081/' \
  /home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
```

网页改为：

```text
http://192.168.25.189:8081/
```

注意：当前 launch 文件不一定支持 `web_port:=8081` 参数覆盖，优先改 yaml。

## 5. `/camera/color/camera_info` 没输出

先查是否有相机节点：

```bash
ros2 node list
```

如果没有：

```text
/direct_realsense
```

说明相机发布节点没启动。重新同步最新版并重编，确认：

```bash
ros2 pkg executables yolo_trt_ros2 | grep direct_realsense
```

如果有 `/direct_realsense`，但 `camera_info` 没输出：

```bash
ros2 topic info -v /camera/color/camera_info
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
ros2 topic hz /camera/color/camera_info
```

## 6. `pyrealsense2 has no attribute context`

报错：

```text
module 'pyrealsense2' has no attribute 'context'
```

说明系统 Python3.10 没有真正的 RealSense binding。检查：

```bash
cd ~/MscapeTech/Foxy_ROS
conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash

python3 -c "import pyrealsense2 as rs; print(rs); print(getattr(rs,'__file__',None)); print('context', hasattr(rs,'context')); print('pipeline', hasattr(rs,'pipeline')); print(dir(rs)[:30])"
python3 -m pip show pyrealsense2
```

如果 `context False`：

```bash
python3 -m pip install --user --force-reinstall pyrealsense2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证：

```bash
python3 -c "import pyrealsense2 as rs; print(rs.__file__); print('context', hasattr(rs,'context')); print('pipeline', hasattr(rs,'pipeline')); print(rs.context().query_devices())"
```

## 7. `PackageNotFoundError: yolo-trt-ros2`

报错：

```text
importlib.metadata.PackageNotFoundError: No package metadata was found for yolo-trt-ros2
```

先确认不要覆盖 `PYTHONPATH`。错误写法：

```bash
export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python
```

正确写法：

```bash
export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH
```

重新编译：

```bash
cd ~/MscapeTech/Foxy_ROS

conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/humble/setup.bash

rm -rf build/yolo_trt_ros2 install/yolo_trt_ros2

colcon build --packages-select detector_msgs yolo_trt_ros2

source install/setup.bash
export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH

python3 -c "import importlib.metadata as m; print(m.distribution('yolo-trt-ros2'))"
python3 -c "import yolo_trt_ros2.coordinate_projector_node; print('coordinate import ok')"
```

如果 launch 子进程仍报 metadata 错误，可以临时绕过 console script，见下一节。

## 8. 分终端手动绕过 `coordinate_projector` 入口脚本

如果一体化 launch 中 `coordinate_projector_node` 因 metadata 报错退出，但相机和 YOLO 已经正常，可以用两个终端。

终端 1：启动相机、YOLO、网页：

```bash
cd ~/MscapeTech/Foxy_ROS

conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/humble/setup.bash
source install/setup.bash

export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch yolo_trt_ros2 inspection_perception.launch.py \
  config_file:=/home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml \
  use_direct_camera:=true
```

保持终端 1 不要关。里面自带的 `coordinate_projector` 如果死掉，先不管。

终端 2：手动启动 projector：

```bash
cd ~/MscapeTech/Foxy_ROS

conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/humble/setup.bash
source install/setup.bash

export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH
export ROS_DOMAIN_ID=42
export ROS_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

python3 -m yolo_trt_ros2.coordinate_projector_node \
  --ros-args \
  -r __node:=coordinate_projector_manual \
  --params-file /home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
```

第三个终端检查：

```bash
ros2 node list
ros2 topic echo --once /detector/objects_3d
ros2 topic echo --once /detector/current_joint_state
ros2 topic echo --once /detector/target_joint_state
```

## 9. `IK skipped: missing current joint values`

说明 3D/检测可能已经在跑，但 projector 没拿到 H2 当前关节值。先测 lowstate：

```bash
cd ~/MscapeTech/Foxy_ROS
source /opt/ros/humble/setup.bash
source install/setup.bash
export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH

python3 -c "import time; from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber; from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_; box={'msg':None}; ChannelFactoryInitialize(0,'eth0'); sub=ChannelSubscriber('rt/lf/lowstate', LowState_); sub.Init(lambda m: box.__setitem__('msg', m), 10); [time.sleep(0.1) for _ in range(30) if box['msg'] is None]; print('lowstate=', 'OK q22='+str(box['msg'].motor_state[22].q) if box['msg'] else 'NO')"
```

如果输出：

```text
lowstate= OK q22=...
```

说明参数是对的，重启 launch 时确保 `export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH` 在 launch 前执行。

如果暂时只想看 3D 坐标，不想刷 IK 警告：

```bash
sed -i 's/publish_target_joint_state: true/publish_target_joint_state: false/' \
  /home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
sed -i 's/publish_objects_ik_json: true/publish_objects_ik_json: false/' \
  /home/unitree/MscapeTech/Foxy_ROS/src/yolo_trt_ros2/config/inspection_perception.yaml
```

## 10. 开门按钮相关 topic

2D 检测：

```bash
ros2 topic echo /detector/objects
```

3D 世界坐标：

```bash
ros2 topic echo /detector/objects_3d
```

更直观的 JSON：

```bash
ros2 topic echo /detector/objects_ik_json
```

蓝色按钮字段含义：

```text
point_world_m              视觉检测到的按钮中心世界坐标
blue_point_contact_world_m 按世界 Z 方向下偏 0.004m 后的实际着力点
ree_target_for_dex1_tip_m  为了让 Dex1 指尖戳到该点，R_ee 应到达的位置
preferred_copy_target_m    推荐给运控/键盘控制直接使用的目标
```

## 11. 直连脚本备用启动

当 ROS 相机链路不稳时，直连脚本仍可用。它运行在 `h1_arm` conda 环境：

```bash
cd ~/MscapeTech/handle_recognition/minimal_test
conda activate h1_arm
export PYTHONPATH=/home/unitree/MscapeTech/unitree_sdk2_python:$PYTHONPATH

python3 run_h2_handle_pose_direct.py \
  --model /home/unitree/MscapeTech/models/yoloe-11s-seg.pt \
  --prompt-free \
  --mobileclip-path "" \
  --handeye /home/unitree/MscapeTech/Hand_Eye_Calib/outputs/eye_in_hand_20260630_150210.json \
  --iface eth0 \
  --lowstate-topic rt/lf/lowstate \
  --fk-backend xr_pinocchio \
  --depth-fallback bbox \
  --print-interval 0.2 \
  --web \
  --web-host 0.0.0.0 \
  --web-port 8081 \
  --dex1-tip-from-wrist 0.14 0.01 0.012 \
  --blue-point-target-world-offset 0 0 -0.004
```

网页：

```text
http://192.168.25.189:8081/
```

Dex1-1 当前现场标定：

```text
--dex1-tip-from-wrist 0.14 0.01 0.012
```

蓝色按钮下沿着力点偏移：

```text
--blue-point-target-world-offset 0 0 -0.004
```

这个偏移只作用于蓝点目标，不改变所有物体的 Dex1 TCP。

