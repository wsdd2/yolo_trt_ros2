# Foxy_ROS third-party bundle

This directory collects the non-ROS-workspace files required by the current
H2 perception configuration. It is intended for engineering handoff and
offline deployment.

## Included source/runtime assets

- `H2_joint_cartesian/`
  - `scripts/h2_xr_official_ik_demo.py`: provides `H2CompatibleIK`.
  - `third_party/xr_teleoperate/`: H2 Pinocchio URDF, meshes and robot control.
  - `third_party/unitree_sdk2_python/`: SDK used by xr_teleoperate.
  - `cyclonedds/install/`: local CycloneDDS install metadata/headers used by
    the H2 integration. The Windows mirror does not contain the compiled
    Linux shared library; install `cyclonedds==0.10.2` on the H2.
- `unitree_sdk2_python/`: SDK path currently bootstrapped directly by
  `coordinate_projector_node.py`.
- `handle_recognition/minimal_test/handle_recognition/`: direct RealSense
  adapter imported by `integrated_perception_node.py` and the legacy
  `direct_realsense_node.py`.
- `robot_kinematics/`: URDF FK/IK fallback backend.
- `unitree_ros/robots/h2_description/`: fallback H2 URDF and meshes.

The duplicated Unitree SDK is intentional: the current code can load the
standalone path, while xr_teleoperate also carries its own vendored copy.

## Required files not available on the development PC

The following runtime files currently exist only on the H2 deployment
environment and must be copied into this bundle before calling it complete:

```text
models/yoloe-11s-seg.pt
models/mobileclip_blt.ts
calibration/eye_in_hand_20260630_150210.json
calibration/eye_in_hand_20260630_150210_npy/T_cam2hand.npy
```

The target's architecture-specific PyTorch/CUDA and CycloneDDS shared
libraries must also be installed on the target; they cannot be copied from
this Windows source tree.

Expected H2 sources:

```text
/home/unitree/MscapeTech/models/yoloe-11s-seg.pt
/home/unitree/MscapeTech/models/mobileclip_blt.ts
/home/unitree/MscapeTech/Hand_Eye_Calib/outputs/eye_in_hand_20260630_150210.json
/home/unitree/MscapeTech/Hand_Eye_Calib/outputs/eye_in_hand_20260630_150210_npy/
```

## System dependencies

- ROS 2 Humble on the current H2 deployment.
- Intel RealSense `librealsense2`.
- NVIDIA/CUDA-compatible PyTorch for the target architecture.
- Packages listed in `apt-packages-humble.txt` and
  `requirements-h2.txt`.

Do not replace ROS Humble's OpenCV with `opencv-python`; `cv_bridge` must use
the system OpenCV ABI.

## Current path mapping

The current YAML/code still references the original deployment layout:

```text
/home/unitree/MscapeTech/models
/home/unitree/MscapeTech/handle_recognition
/home/unitree/MscapeTech/unitree_sdk2_python
/home/unitree/H2_joint_cartesian
/home/unitree/MscapeTech/robot_kinematics
/home/unitree/MscapeTech/unitree_ros
```

This bundle is therefore a dependency handoff, not yet a relocation of all
runtime paths. Preserve upstream license files when distributing it.
