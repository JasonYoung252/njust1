# 依赖安装

适用环境：Ubuntu 20.04、ROS Noetic、Python 3.8。

## ROS 与系统依赖

首次使用 rosdep 时初始化一次，然后安装整个工作空间声明的依赖：

```bash
sudo rosdep init
rosdep update
source /opt/ros/noetic/setup.bash
cd /home/nlg/Arm_ws
rosdep install --from-paths src --ignore-src --rosdistro noetic -r -y
```

已经初始化 rosdep 的机器跳过 `sudo rosdep init`。

## 视觉模块

`package.xml` 声明可由 rosdep 安装的 NumPy、SciPy 和 OpenCV。PaddleOCR、PaddlePaddle 与 RealSense Python SDK 使用 requirements.txt 管理。

建议在可访问 ROS 系统包的独立环境中安装，避免覆盖系统 Python：

```bash
python3 -m venv --system-site-packages /home/nlg/Arm_ws/.venv
source /home/nlg/Arm_ws/.venv/bin/activate
python -m pip install -r /home/nlg/Arm_ws/src/njust/requirements.txt
python -m pip check
```

虚拟环境需要系统安装 `python3-venv`。requirements 固定了当前环境的大部分版本；OpenCV 使用 4.10.0.84，以配合 NumPy 1.24.4，该完整组合尚需相机与 OCR 实测。

OpenCV 选择带 GUI 的 contrib 版本，以支持代码中的 `cv2.imshow`。同一环境避免同时安装多个 OpenCV 发行包；PaddleOCR 的间接依赖也可能引入其他 OpenCV 发行包，安装后须检查。独立环境不会自动隔离通过 `--system-site-packages` 可见的用户级包。

PaddleOCR 首次创建实例可能下载识别模型，需要网络或预先准备模型缓存。Python SDK 可导入不等于相机 USB 权限及采集配置已验证。

可选视觉示例还需要：

```bash
python -m pip install -r /home/nlg/Arm_ws/src/rm_demo/requirements.txt
```

ROS 节点必须使用安装依赖的同一 Python 解释器。现有脚本的 `#!/usr/bin/env python3` 会受 PATH 影响，启动前先激活环境并 source ROS 与工作空间。

## 编译与检查

```bash
source /opt/ros/noetic/setup.bash
cd /home/nlg/Arm_ws
catkin build
source devel/setup.bash
rosdep check --from-paths src --ignore-src --rosdistro noetic
```

依赖补全不会修复 `arm_control_nl.py` 与当前服务字段不匹配的问题。
