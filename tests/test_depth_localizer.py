# -*- coding: utf-8 -*-
import math

import numpy as np
import pytest

import pose_utils
from depth_localizer import (STATUS_FAIL, STATUS_PLANE, STATUS_ROI_MEDIAN, CameraIntrinsics, DepthLocalizer,
                             LocalizerConfig, box_from_polygon, facing_yaw_from_normal, fit_plane_ransac,
                             fuse_depth_frames, robust_roi_depth)
from synthetic_d435 import SceneParams, SyntheticScene


def test_status_codes_match_pose_utils():
    assert (STATUS_FAIL, STATUS_PLANE, STATUS_ROI_MEDIAN) == (
        pose_utils.STATUS_FAIL, pose_utils.STATUS_PLANE, pose_utils.STATUS_ROI_MEDIAN)


def test_fuse_ignores_zeros_and_rejects_inconsistent_pixels():
    stack = np.zeros((5, 2, 3), dtype=np.float32)
    stack[:, 0, 0] = [0.35, 0, 0.351, 0.349, 0]          # 3 帧有效 -> 中位数
    stack[:, 0, 1] = [0.35, 0, 0, 0, 0]                  # 只有 1 帧有效 -> 0
    stack[:, 0, 2] = [0.30, 0.40, 0.31, 0.42, 0.29]      # 离散度过大 -> 0
    fused = fuse_depth_frames(stack, min_valid_count=2, max_spread_m=0.006)
    assert fused[0, 0] == pytest.approx(0.35, abs=1e-6)
    assert fused[0, 1] == 0
    assert fused[0, 2] == 0


def test_roi_depth_robust_to_outliers_and_holes():
    cfg = LocalizerConfig()
    d = np.full((40, 40), 0.35, dtype=np.float32)
    d[::2, ::3] = 0                 # 空洞
    d[5:8, 5:8] = 0.38              # 少量错误深度
    z, sigma, ratio, n = robust_roi_depth(d, (0, 0, 39, 39), cfg)
    assert z == pytest.approx(0.35, abs=1e-4)
    assert 0.7 < ratio < 0.9


def test_roi_depth_fails_when_mostly_holes():
    cfg = LocalizerConfig()
    d = np.zeros((40, 40), dtype=np.float32)
    d[0, :5] = 0.35
    z, _, ratio, _ = robust_roi_depth(d, (0, 0, 39, 39), cfg)
    assert z is None and ratio < 0.05


def test_plane_ransac_recovers_plane_with_outliers():
    rng = np.random.default_rng(1)
    n_true = np.array([0.1, -0.05, -1.0])
    n_true /= np.linalg.norm(n_true)
    p0 = np.array([0.0, 0.0, 0.35])
    a = np.cross(n_true, [1, 0, 0]); a /= np.linalg.norm(a)
    b = np.cross(n_true, a)
    s = rng.uniform(-0.05, 0.05, (800, 2))
    pts = p0 + s[:, :1] * a + s[:, 1:] * b + rng.normal(0, 0.0004, (800, 1)) * n_true
    outliers = p0 + rng.uniform(-0.05, 0.05, (200, 3))
    plane = fit_plane_ransac(np.vstack([pts, outliers]), 0.002, 200, rng)
    assert plane is not None
    assert np.degrees(np.arccos(abs(plane.normal @ n_true))) < 0.5
    assert plane.normal[2] < 0          # 朝向相机
    assert plane.inlier_ratio > 0.75


@pytest.mark.parametrize("label_dropout", [0.0, 0.6, 1.0])
def test_localize_on_synthetic_scene(label_dropout):
    rng = np.random.default_rng(7)
    p = SceneParams(distance_m=0.35, yaw_deg=6.0, label_dropout=label_dropout, plate_dropout=0.95,
                    label_offset_px=(60.0, 30.0))
    scene = SyntheticScene(p, rng)
    frames = scene.render_frames(5)
    K = CameraIntrinsics(p.fx, p.fy, p.cx, p.cy)
    poly = scene.label_polygon()
    center = np.mean(np.array(poly), axis=0)
    res = DepthLocalizer(K).localize(frames, (int(center[0]), int(center[1])), box_from_polygon(poly))
    assert res.ok and res.status == STATUS_PLANE, res.message
    err = np.linalg.norm(res.point_cam - scene.label_center) * 1000
    assert err < 2.0, err
    ang = np.degrees(np.arccos(np.clip(res.normal_cam @ scene.normal, -1, 1)))
    assert ang < 1.0, ang
    assert 1.0 < res.sigma_normal_m * 1000 < 5.0


def test_localize_fails_cleanly_without_depth():
    K = CameraIntrinsics(700, 700, 480, 270)
    frames = np.zeros((3, 540, 960), dtype=np.float32)
    res = DepthLocalizer(K).localize(frames, (480, 270), (470, 265, 490, 275))
    assert not res.ok and res.status == STATUS_FAIL and res.point_cam is None


def test_localize_center_outside_image():
    K = CameraIntrinsics(700, 700, 480, 270)
    res = DepthLocalizer(K).localize(np.ones((540, 960), np.float32) * 0.35, (2000, 10))
    assert not res.ok


def test_roi_fallback_when_plane_unavailable():
    # 只有标签附近一小块有深度：平面点数不足，应回退到 ROI 中位数
    K = CameraIntrinsics(700, 700, 480, 270)
    d = np.zeros((540, 960), dtype=np.float32)
    d[262:279, 465:496] = 0.35
    res = DepthLocalizer(K).localize(d, (480, 270), (465, 262, 495, 278))
    assert res.ok and res.status == STATUS_ROI_MEDIAN
    assert res.point_cam[2] == pytest.approx(0.35, abs=1e-4)


def test_facing_yaw_from_normal():
    # 面板法向指向 -x（朝向位于原点的机械臂），说明机械臂面朝 +x，方向角 0
    assert facing_yaw_from_normal(np.array([-1.0, 0, 0])) == pytest.approx(0.0)
    assert math.degrees(facing_yaw_from_normal(np.array([0, 1.0, 0]))) == pytest.approx(-90.0)
