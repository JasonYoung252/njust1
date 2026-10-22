#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/pose 服务返回值的打包与解析，以及“偏移量随面板偏航修正”的计算。只依赖标准库，方便单元测试。

/pose 返回 float32[] pose。前 7 个元素与原程序完全一致，原有调用方不受影响：
    [0:3]  x, y, z              标签在机械臂基座坐标系下的位置（m）
    [3:7]  qx, qy, qz, qw       当前末端姿态（四元数，沿用原逻辑）
新增 6 个元素（长度 13）：
    [7]    facing_yaw           机械臂“面朝面板”的实测方向角（rad，基座坐标系，0 = 面朝 +x）；无法估计时为 NaN
    [8]    sigma_normal         沿面板法向的定位标准差（m）
    [9]    sigma_lateral        横向定位标准差（m）
    [10]   roi_valid_ratio      标签 ROI 的有效深度比例（0–1）
    [11]   plane_inlier_ratio   面板平面内点比例（0–1）
    [12]   status               0 失败，1 面板平面，2 标签 ROI 中位数
定位失败时返回 13 个 0。原程序的有效性判断（目标比末端离基座更远）会把它判为无效。
"""
import math

POSE_LEN = 13
IDX_FACING_YAW = 7
IDX_SIGMA_NORMAL = 8
IDX_SIGMA_LATERAL = 9
IDX_ROI_RATIO = 10
IDX_PLANE_RATIO = 11
IDX_STATUS = 12

STATUS_FAIL = 0
STATUS_PLANE = 1
STATUS_ROI_MEDIAN = 2


def failed_pose():
    return [0.0] * POSE_LEN


def pack_pose(position, quaternion, facing_yaw=float("nan"), sigma_normal=float("nan"),
              sigma_lateral=float("nan"), roi_valid_ratio=0.0, plane_inlier_ratio=0.0, status=STATUS_FAIL):
    out = [float(position[0]), float(position[1]), float(position[2]),
           float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]),
           float(facing_yaw), float(sigma_normal), float(sigma_lateral),
           float(roi_valid_ratio), float(plane_inlier_ratio), float(status)]
    return out


def unpack_pose(pose):
    """
    解析 /pose 返回值，兼容旧版只有 7 个元素的返回。
    返回 dict：x, y, z, quat, facing_yaw, sigma_normal, sigma_lateral, roi_valid_ratio,
    plane_inlier_ratio, status, extended, valid。
    """
    p = [float(a) for a in pose]
    if len(p) < 7:
        p = p + [0.0] * (7 - len(p))
    info = {
        "x": p[0], "y": p[1], "z": p[2],
        "quat": p[3:7],
        "extended": len(p) >= POSE_LEN,
        "facing_yaw": float("nan"),
        "sigma_normal": float("nan"),
        "sigma_lateral": float("nan"),
        "roi_valid_ratio": float("nan"),
        "plane_inlier_ratio": float("nan"),
        "status": None,
    }
    if info["extended"]:
        info["facing_yaw"] = p[IDX_FACING_YAW]
        info["sigma_normal"] = p[IDX_SIGMA_NORMAL]
        info["sigma_lateral"] = p[IDX_SIGMA_LATERAL]
        info["roi_valid_ratio"] = p[IDX_ROI_RATIO]
        info["plane_inlier_ratio"] = p[IDX_PLANE_RATIO]
        info["status"] = int(round(p[IDX_STATUS]))
        info["valid"] = info["status"] in (STATUS_PLANE, STATUS_ROI_MEDIAN)
    else:
        info["valid"] = any(abs(a) > 1e-9 for a in p[:3])
    return info


def wrap_angle(a):
    """把角度归一化到 (-pi, pi]。"""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0:
        a += 2.0 * math.pi
    return a - math.pi


def nominal_facing_yaw(direction):
    """
    arm_control 中 direction（前右后左 = 0123，个位有效）对应的名义朝向角。
    与原 dxy() 一致：direction=0 面朝 +x，direction=1 面朝 -y。
    """
    a = int(direction) % 10
    return -a * math.pi / 2.0


def yaw_correction(direction, facing_yaw, max_correction_deg=10.0):
    """
    实测朝向与名义朝向之差 delta（rad）。超过上限或无效时返回 (0.0, False)，表示不修正。
    """
    if facing_yaw is None or not math.isfinite(facing_yaw):
        return 0.0, False
    delta = wrap_angle(facing_yaw - nominal_facing_yaw(direction))
    if abs(delta) > math.radians(max_correction_deg):
        return 0.0, False
    return delta, True


def rotate_offset(x, y, direction, yaw_delta=0.0):
    """
    把“面朝标签”坐标系下的平面偏移 (x 前, y 左) 转到基座坐标系，等价于原 dxy()，
    并可叠加实测偏航 yaw_delta。yaw_delta=0 时结果与原程序完全相同。
    """
    a = int(direction) % 10
    theta = math.pi * a / 2.0 - yaw_delta
    dx = x * math.cos(theta) + y * math.sin(theta)
    dy = -x * math.sin(theta) + y * math.cos(theta)
    return dx, dy
