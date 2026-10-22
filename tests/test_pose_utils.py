# -*- coding: utf-8 -*-
import math
import subprocess
import os

import numpy as np
import pytest

import pose_utils
import convert


def original_dxy(x, y, direction):
    """原 arm_control.dxy 的实现（逐字复制）。"""
    a = direction % 10
    dx = x * np.cos(np.pi*a/2) + y * np.sin(np.pi*a/2)
    dy = - x * np.sin(np.pi*a/2) + y * np.cos(np.pi*a/2)
    return dx, dy


@pytest.mark.parametrize("direction", [0, 1, 2, 3, 10, 11, 12, 13])
def test_rotate_offset_matches_original_when_no_yaw(direction):
    rng = np.random.default_rng(direction)
    for x, y in rng.uniform(-0.2, 0.2, (20, 2)):
        assert pose_utils.rotate_offset(x, y, direction, 0.0) == pytest.approx(original_dxy(x, y, direction), abs=1e-12)


@pytest.mark.parametrize("direction", [0, 1, 2, 3])
def test_yaw_correction_points_forward_offset_along_measured_facing(direction):
    measured = pose_utils.nominal_facing_yaw(direction) + math.radians(6.0)
    delta, ok = pose_utils.yaw_correction(direction, measured, 10.0)
    assert ok and delta == pytest.approx(math.radians(6.0))
    dx, dy = pose_utils.rotate_offset(1.0, 0.0, direction, delta)
    assert math.atan2(dy, dx) == pytest.approx(pose_utils.wrap_angle(measured), abs=1e-9)


def test_yaw_correction_rejects_large_or_missing():
    assert pose_utils.yaw_correction(0, math.radians(25), 10.0) == (0.0, False)
    assert pose_utils.yaw_correction(0, float("nan"), 10.0) == (0.0, False)
    assert pose_utils.yaw_correction(0, None, 10.0) == (0.0, False)


def test_pack_unpack_roundtrip_and_legacy():
    packed = pose_utils.pack_pose((0.5, 0.1, 0.3), (0, 0, 0, 1), 0.05, 0.002, 0.001, 0.4, 0.8, 1)
    assert len(packed) == pose_utils.POSE_LEN
    info = pose_utils.unpack_pose(packed)
    assert info["valid"] and info["extended"] and info["status"] == 1
    assert info["facing_yaw"] == pytest.approx(0.05)
    legacy = pose_utils.unpack_pose([0.5, 0.1, 0.3, 0, 0, 0, 1])
    assert legacy["valid"] and not legacy["extended"]
    assert not pose_utils.unpack_pose(pose_utils.failed_pose())["valid"]
    assert not pose_utils.unpack_pose([0] * 7)["valid"]


def test_convert_identical_to_original():
    """新版 convert 与仓库原版逐点一致（从 git HEAD 取原文件执行）。"""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        src = subprocess.check_output(["git", "-C", root, "show", "HEAD:scripts/convert.py"], text=True)
    except Exception:
        pytest.skip("没有 git 历史")
    ns = {}
    exec(compile(src, "convert_orig", "exec"), ns)
    rng = np.random.default_rng(3)
    for _ in range(50):
        cam = rng.uniform(-0.3, 0.5, 3)
        pose = list(rng.uniform(-0.5, 0.5, 3)) + list(rng.uniform(-3.1, 3.1, 3))
        a = ns["convert"](*cam, *pose)
        b = convert.convert(*cam, *pose)
        assert np.allclose(a, b, atol=1e-12)


def test_rotate_to_base_consistent_with_convert():
    rng = np.random.default_rng(4)
    pose = list(rng.uniform(-0.5, 0.5, 3)) + list(rng.uniform(-3.1, 3.1, 3))
    p1 = np.array(convert.convert(0.01, 0.02, 0.3, *pose))
    p2 = np.array(convert.convert(0.01, 0.02, 0.4, *pose))
    v = convert.rotate_to_base([0, 0, 0.1], *pose)
    assert np.allclose(p2 - p1, v, atol=1e-12)
