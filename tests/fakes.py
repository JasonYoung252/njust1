# -*- coding: utf-8 -*-
"""
没有 ROS、相机和 OCR 时，用假模块替代 rospy / pyrealsense2 / paddleocr / rm_msgs / njust 消息，
让 target_position.py、D435_rgb_depth.py、arm_control.py 的真实代码在仿真数据上跑通。
"""
import sys
import types

import numpy as np


# ----------------------------------------------------------------------------------------------
# rospy
# ----------------------------------------------------------------------------------------------
class FakeRospy(types.ModuleType):
    class ROSInterruptException(Exception):
        pass

    def __init__(self):
        super().__init__("rospy")
        self.params = {}
        self.logs = []
        self.published = []
        self.service_handler = None

    def init_node(self, *a, **k):
        pass

    def get_param(self, name, default=None):
        return self.params.get(name, default)

    def has_param(self, name):
        return name in self.params

    def loginfo(self, msg):
        self.logs.append(("info", msg))

    def logwarn(self, msg):
        self.logs.append(("warn", msg))

    def logerr(self, msg):
        self.logs.append(("error", msg))

    def sleep(self, *a):
        pass

    def is_shutdown(self):
        return False

    def wait_for_service(self, *a, **k):
        pass

    def spin(self):
        pass

    def Publisher(self, topic, *a, **k):
        outer = self

        class _Pub:
            def publish(self, msg):
                outer.published.append((topic, msg))
        return _Pub()

    def Subscriber(self, *a, **k):
        return object()

    def Service(self, *a, **k):
        return object()

    def ServiceProxy(self, *a, **k):
        outer = self

        def call(req):
            return outer.service_handler(req)
        return call


# ----------------------------------------------------------------------------------------------
# 消息与服务
# ----------------------------------------------------------------------------------------------
class _Msg:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if args:
            self.args = args


class PoseResponse:
    def __init__(self, pose=None):
        self.pose = list(pose) if pose is not None else []


def _msg_module(name, classes):
    mod = types.ModuleType(name)
    for c in classes:
        setattr(mod, c, type(c, (_Msg,), {}))
    return mod


# ----------------------------------------------------------------------------------------------
# pyrealsense2：用 SyntheticScene 生成的深度驱动 D435_rgb_depth.RealSenseProcessor
# ----------------------------------------------------------------------------------------------
class _Intr:
    def __init__(self, fx, fy, cx, cy, w, h):
        self.fx, self.fy, self.ppx, self.ppy, self.width, self.height = fx, fy, cx, cy, w, h


class _Profile:
    def __init__(self, intr):
        self.intrinsics = intr

    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return self.intrinsics


class _Frame:
    def __init__(self, data, intr, scale=0.001):
        self._data = data
        self.profile = _Profile(intr)
        self._scale = scale

    def get_data(self):
        return self._data

    def get_distance(self, x, y):
        return float(self._data[int(y), int(x)]) * self._scale

    def __bool__(self):
        return True

    def as_depth_frame(self):
        return self

    def as_frameset(self):
        raise RuntimeError("not a frameset")


class _Frameset:
    def __init__(self, depth, color):
        self._depth, self._color = depth, color

    def get_depth_frame(self):
        return self._depth

    def get_color_frame(self):
        return self._color

    def as_frameset(self):
        return self

    def __bool__(self):
        return True


class FakeRealsense(types.ModuleType):
    """frame_source() 返回 (depth_m(H,W), color(H,W,3))。"""

    def __init__(self):
        super().__init__("pyrealsense2")
        self.frame_source = None
        self.intr = None
        rs = self

        class stream:
            color = "color"
            depth = "depth"

        class format:
            bgr8 = "bgr8"
            z16 = "z16"

        class option:
            emitter_enabled = "emitter_enabled"
            laser_power = "laser_power"
            visual_preset = "visual_preset"
            min_distance = "min_distance"
            max_distance = "max_distance"

        class rs400_visual_preset:
            default = 1
            high_accuracy = 3
            high_density = 4

        class _Range:
            min, max = 0.0, 360.0

        class _Sensor:
            def __init__(self):
                self.options = {}

            def get_depth_scale(self):
                return 0.001

            def supports(self, opt):
                return True

            def set_option(self, opt, val):
                self.options[opt] = val

            def get_option_range(self, opt):
                return _Range()

        class _Device:
            def __init__(self):
                self.sensor = _Sensor()

            def first_depth_sensor(self):
                return self.sensor

        class _PipelineProfile:
            def __init__(self):
                self.device = _Device()

            def get_device(self):
                return self.device

            def get_stream(self, s):
                return _Profile(rs.intr)

        class pipeline:
            def __init__(self):
                self.prof = _PipelineProfile()
                self.frames_served = 0

            def start(self, cfg):
                return self.prof

            def wait_for_frames(self):
                self.frames_served += 1
                depth_m, color = rs.frame_source()
                depth_raw = np.round(depth_m / 0.001).astype(np.uint16)
                return _Frameset(_Frame(depth_raw, rs.intr), _Frame(color, rs.intr))

            def stop(self):
                pass

        class config:
            def enable_stream(self, *a):
                pass

        class align:
            def __init__(self, s):
                pass

            def process(self, frames):
                return frames

        class _Filter:
            def __init__(self, *a):
                pass

            def set_option(self, *a):
                pass

            def process(self, f):
                return f

        def rs2_deproject_pixel_to_point(intr, pixel, depth):
            x = (pixel[0] - intr.ppx) / intr.fx * depth
            y = (pixel[1] - intr.ppy) / intr.fy * depth
            return [x, y, depth]

        self.stream, self.format, self.option = stream, format, option
        self.rs400_visual_preset = rs400_visual_preset
        self.pipeline, self.config, self.align = pipeline, config, align
        self.threshold_filter = self.disparity_transform = self.spatial_filter = _Filter
        self.rs2_deproject_pixel_to_point = rs2_deproject_pixel_to_point
        self.intrinsics = _Intr


# ----------------------------------------------------------------------------------------------
# PaddleOCR
# ----------------------------------------------------------------------------------------------
class FakePaddleOCRModule(types.ModuleType):
    def __init__(self):
        super().__init__("paddleocr")
        self.results = [[]]
        outer = self

        class PaddleOCR:
            def __init__(self, *a, **k):
                pass

            def ocr(self, img, cls=True):
                return outer.results
        self.PaddleOCR = PaddleOCR


def install():
    """安装所有假模块，返回 dict 便于测试中控制。"""
    rospy = FakeRospy()
    rs = FakeRealsense()
    paddle = FakePaddleOCRModule()
    njust = types.ModuleType("njust")
    njust_srv = types.ModuleType("njust.srv")
    for name in ("task", "taskResponse", "pose", "poseRequest"):
        setattr(njust_srv, name, type(name, (_Msg,), {}))
    njust_srv.poseResponse = PoseResponse
    njust_msg = types.ModuleType("njust.msg")
    njust_msg.result = type("result", (_Msg,), {})
    rm_msgs = types.ModuleType("rm_msgs")
    rm_msgs_msg = _msg_module("rm_msgs.msg", ["Arm_Current_State", "ArmState", "GetArmState_Command", "Gripper_Pick",
                                             "Gripper_Set", "MoveJ", "MoveJ_P", "MoveL", "Plan_State"])
    robotic_arm = types.ModuleType("robotic_arm")
    robotic_arm.RML63_II = 63
    robotic_arm.__all__ = ["RML63_II"]
    sys.modules.update({
        "rospy": rospy, "pyrealsense2": rs, "paddleocr": paddle,
        "njust": njust, "njust.srv": njust_srv, "njust.msg": njust_msg,
        "rm_msgs": rm_msgs, "rm_msgs.msg": rm_msgs_msg, "robotic_arm": robotic_arm,
    })
    return {"rospy": rospy, "rs": rs, "paddle": paddle}
