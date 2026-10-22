#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
压板 / 标签的鲁棒深度定位（纯 numpy，不依赖 ROS 和 pyrealsense2，可离线仿真与单元测试）

背景：D435 在压板的金属片、光滑外壳上经常出现深度空洞。原程序只读取检测中心一个像素的深度，
读到 0 时会得到相机原点 (0, 0, 0)，而且不报错。本模块按下面的顺序定位：

    1. 多帧融合：逐像素取多帧有效深度的中位数，忽略 0 值。
    2. ROI 鲁棒取值：在标签文字框内取有效深度的中位数，用 MAD 剔除离群值，并检查有效像素比例。
    3. 面板平面先验（主方法）：标签贴在屏柜面板上，面板面积大、深度有效率高。
       在标签周围用 RANSAC 拟合面板平面，再求“标签像素射线”与平面的交点。
       即使标签和压板本身全是空洞，也能定位，同时得到面板法向。
    4. 不确定性：输出沿法向和横向的标准差（m）、ROI 有效率、平面内点比例，供上层判断和鲁棒优化使用。

坐标约定：相机坐标系 x 向右、y 向下、z 向前（与 RealSense 一致），单位 m。
深度图：二维数组，单位 m，0 表示无效。
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# 定位状态码（也写入 /pose 服务返回值的第 13 个元素）
STATUS_FAIL = 0
STATUS_PLANE = 1        # 面板平面 + 射线求交
STATUS_ROI_MEDIAN = 2   # 标签 ROI 中位数（平面不可用时的后备）

STATUS_NAMES = {STATUS_FAIL: "fail", STATUS_PLANE: "plane", STATUS_ROI_MEDIAN: "roi_median"}


@dataclass
class CameraIntrinsics:
    """针孔内参（像素）。对齐到彩色图后的深度，使用彩色相机内参。"""
    fx: float
    fy: float
    cx: float
    cy: float

    def ray(self, u: float, v: float) -> np.ndarray:
        """像素 (u, v) 的视线方向，z 分量为 1。"""
        return np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])


@dataclass
class LocalizerConfig:
    """定位参数。默认值按 D435、30–40 cm 工作距离设置，现场需要用实测数据再标定。"""
    # 标签 ROI
    roi_pad_px: int = 0                  # 文字框向外扩展的像素数，负数表示收缩
    min_roi_valid_ratio: float = 0.15    # ROI 内有效深度比例下限
    min_roi_valid_px: int = 15           # ROI 内有效像素个数下限
    max_roi_mad_m: float = 0.004         # ROI 深度离散度（MAD 换算的 σ）上限
    # 多帧融合
    frame_min_valid_count: int = 2       # 像素至少在几帧中有效才保留
    frame_max_spread_m: float = 0.006    # 同一像素多帧之间的离散度上限
    # 面板平面
    plane_radius_m: float = 0.06         # 在标签周围多大半径内取点拟合平面
    plane_depth_gate_m: float = 0.03     # 只用深度接近参考深度的点，排除背景和近处遮挡
    plane_inlier_thresh_m: float = 0.003 # RANSAC 内点阈值
    plane_iters: int = 200               # RANSAC 迭代次数
    plane_min_points: int = 150          # 参与拟合的最少点数
    plane_min_inlier_ratio: float = 0.4  # 内点比例下限
    plane_max_points: int = 4000         # 参与拟合的最多点数（超过则随机降采样）
    plane_min_extent_m: float = 0.006    # 内点在平面内较窄方向的标准差下限，点集太窄时法向不可信
    max_ray_plane_angle_deg: float = 60.0  # 视线与面板法向夹角上限，过于倾斜时不可信
    max_plane_roi_disagreement_m: float = 0.012  # 平面结果与标签 ROI 中位数允许的最大差异
    # 不确定性与有效范围
    pixel_sigma_px: float = 2.0          # 检测中心的像素抖动（标准差）
    systematic_sigma_m: float = 0.0015   # 深度系统误差（标定偏差、温漂等），按经验给定，需用真机数据标定
    min_depth_m: float = 0.10
    max_depth_m: float = 1.50
    random_seed: Optional[int] = 0       # 固定随机种子，便于复现


@dataclass
class PlaneFit:
    normal: np.ndarray       # 单位法向，指向相机
    d: float                 # 平面方程 n·p + d = 0
    centroid: np.ndarray
    rms: float               # 内点残差 RMS（m）
    spread: float            # 内点在平面内分布的尺度（m），用于估计倾角误差
    spread_min: float        # 内点在平面内较窄方向的标准差（m）
    n_points: int
    n_inliers: int

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / float(max(self.n_points, 1))


@dataclass
class LocalizationResult:
    ok: bool
    status: int
    point_cam: Optional[np.ndarray]       # 标签中心在相机坐标系下的位置（m）
    normal_cam: Optional[np.ndarray]      # 面板法向（指向相机），只有平面方法给出
    sigma_normal_m: float                 # 沿面板法向（近似为深度方向）的标准差
    sigma_lateral_m: float                # 横向标准差（来自像素抖动）
    roi_valid_ratio: float                # 标签 ROI 的有效深度比例
    plane_inlier_ratio: float
    n_plane_inliers: int
    roi_depth_m: Optional[float]          # 标签 ROI 中位深度（可能为 None）
    message: str

    @property
    def method(self) -> str:
        return STATUS_NAMES.get(self.status, "unknown")


# ----------------------------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------------------------
def box_from_polygon(polygon: Sequence[Sequence[float]]) -> Tuple[int, int, int, int]:
    """OCR 四点框 -> 轴对齐框 (u0, v0, u1, v1)，闭区间。"""
    pts = np.asarray(polygon, dtype=float).reshape(-1, 2)
    u0, v0 = np.floor(pts.min(axis=0)).astype(int)
    u1, v1 = np.ceil(pts.max(axis=0)).astype(int)
    return int(u0), int(v0), int(u1), int(v1)


def clip_box(box: Tuple[int, int, int, int], shape: Tuple[int, int], pad: int = 0) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    u0, v0, u1, v1 = box
    u0, v0, u1, v1 = u0 - pad, v0 - pad, u1 + pad, v1 + pad
    if u1 < u0:
        u0 = u1 = (u0 + u1) // 2
    if v1 < v0:
        v0 = v1 = (v0 + v1) // 2
    return max(0, u0), max(0, v0), min(w - 1, u1), min(h - 1, v1)


def fuse_depth_frames(frames, min_valid_count: int = 2, max_spread_m: Optional[float] = 0.006) -> np.ndarray:
    """
    多帧深度逐像素融合。

    frames: (N, H, W) 或 (H, W)，单位 m，0 为无效。
    返回 (H, W) 的融合深度：每个像素取有效值的中位数；有效帧数不足或多帧离散度过大时置 0。
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        return arr.copy()
    n = arr.shape[0]
    if n == 1:
        return arr[0].copy()
    valid = arr > 0
    count = valid.sum(axis=0)
    masked = np.where(valid, arr, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(masked, axis=0)
        keep = count >= min(min_valid_count, n)
        if max_spread_m is not None:
            mad = np.nanmedian(np.abs(masked - med[None]), axis=0) * 1.4826
            keep &= ~(mad > max_spread_m)
    fused = np.where(keep & np.isfinite(med), med, 0.0).astype(np.float32)
    return fused


def robust_roi_depth(depth: np.ndarray, box: Tuple[int, int, int, int], cfg: LocalizerConfig):
    """
    ROI 内的鲁棒深度。

    返回 (z_median, sigma, valid_ratio, n_valid)。有效像素不足或离散度过大时 z_median 为 None。
    """
    u0, v0, u1, v1 = clip_box(box, depth.shape, cfg.roi_pad_px)
    crop = depth[v0:v1 + 1, u0:u1 + 1]
    total = crop.size
    vals = crop[(crop > cfg.min_depth_m) & (crop < cfg.max_depth_m)].astype(np.float64)
    ratio = vals.size / float(max(total, 1))
    if vals.size < cfg.min_roi_valid_px or ratio < cfg.min_roi_valid_ratio:
        return None, float("nan"), ratio, int(vals.size)
    med = float(np.median(vals))
    sigma = 1.4826 * float(np.median(np.abs(vals - med)))
    # 第二轮：剔除离群值后重新估计（门限至少 2 mm，避免噪声极小时把正常点也剔掉）
    gate = max(3.0 * sigma, 0.002)
    vals = vals[np.abs(vals - med) <= gate]
    if vals.size < cfg.min_roi_valid_px:
        return None, sigma, ratio, int(vals.size)
    med = float(np.median(vals))
    sigma = 1.4826 * float(np.median(np.abs(vals - med)))
    if sigma > cfg.max_roi_mad_m:
        return None, sigma, ratio, int(vals.size)
    return med, sigma, ratio, int(vals.size)


def deproject(us: np.ndarray, vs: np.ndarray, zs: np.ndarray, K: CameraIntrinsics) -> np.ndarray:
    """像素 + 深度 -> 相机坐标系三维点，返回 (N, 3)。"""
    x = (us - K.cx) / K.fx * zs
    y = (vs - K.cy) / K.fy * zs
    return np.stack([x, y, zs], axis=-1)


def fit_plane_ransac(points: np.ndarray, thresh: float, iters: int, rng: np.random.Generator) -> Optional[PlaneFit]:
    """RANSAC 拟合平面，再用内点做最小二乘（SVD）精修。法向朝向相机（原点）。"""
    n = points.shape[0]
    if n < 3:
        return None
    best_mask = None
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm)
        if norm < 1e-12:
            continue
        nrm = nrm / norm
        dist = np.abs(points @ nrm - nrm @ p0)
        mask = dist < thresh
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count < 3:
        return None

    # 两轮最小二乘精修
    mask = best_mask
    for _ in range(2):
        inl = points[mask]
        centroid = inl.mean(axis=0)
        _, s, vt = np.linalg.svd(inl - centroid, full_matrices=False)
        nrm = vt[-1]
        dist = np.abs((points - centroid) @ nrm)
        new_mask = dist < thresh
        if new_mask.sum() < 3:
            break
        mask = new_mask

    inl = points[mask]
    centroid = inl.mean(axis=0)
    _, s, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    nrm = vt[-1]
    if np.dot(nrm, -centroid) < 0:   # 法向朝向相机
        nrm = -nrm
    d = -float(nrm @ centroid)
    res = (inl - centroid) @ nrm
    rms = float(np.sqrt(np.mean(res ** 2)))
    # 内点在平面内的分布尺度：两个较大奇异值对应的标准差的平均
    spread = float(np.mean(s[:2]) / math.sqrt(max(inl.shape[0], 1)))
    spread_min = float(s[1] / math.sqrt(max(inl.shape[0], 1)))
    return PlaneFit(normal=nrm, d=d, centroid=centroid, rms=rms, spread=max(spread, 1e-6),
                    spread_min=spread_min, n_points=n, n_inliers=int(inl.shape[0]))


def intersect_ray_plane(ray: np.ndarray, plane: PlaneFit) -> Optional[np.ndarray]:
    denom = float(plane.normal @ ray)
    if abs(denom) < 1e-9:
        return None
    t = -plane.d / denom
    if t <= 0:
        return None
    return t * ray


# ----------------------------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------------------------
class DepthLocalizer:
    def __init__(self, intrinsics: CameraIntrinsics, config: Optional[LocalizerConfig] = None):
        self.K = intrinsics
        self.cfg = config or LocalizerConfig()
        self.rng = np.random.default_rng(self.cfg.random_seed)

    # 只融合需要的区域，避免对整幅 1080p 图做多帧中位数
    def _crop_window(self, shape, u: float, v: float) -> Tuple[int, int, int, int]:
        r = int(math.ceil(self.K.fx * self.cfg.plane_radius_m / max(self.cfg.min_depth_m, 0.2))) + 2
        h, w = shape
        return max(0, int(u) - r), max(0, int(v) - r), min(w - 1, int(u) + r), min(h - 1, int(v) + r)

    def localize(self, depth_frames, center_uv: Tuple[float, float],
                 box: Optional[Tuple[int, int, int, int]] = None) -> LocalizationResult:
        """
        depth_frames: (N, H, W) 或 (H, W)，单位 m，已对齐到检测所用的彩色图。
        center_uv:   标签中心像素。
        box:         标签框 (u0, v0, u1, v1)；为空时用中心周围 9×9 的小窗口。
        """
        cfg = self.cfg
        arr = np.asarray(depth_frames, dtype=np.float32)
        shape = arr.shape[-2:]
        u, v = float(center_uv[0]), float(center_uv[1])
        if not (0 <= u < shape[1] and 0 <= v < shape[0]):
            return self._fail("检测中心超出图像范围", 0.0)

        # 1) 只在局部窗口内做多帧融合
        wu0, wv0, wu1, wv1 = self._crop_window(shape, u, v)
        if arr.ndim == 3:
            win = fuse_depth_frames(arr[:, wv0:wv1 + 1, wu0:wu1 + 1], cfg.frame_min_valid_count, cfg.frame_max_spread_m)
        else:
            win = arr[wv0:wv1 + 1, wu0:wu1 + 1].copy()
        uc, vc = u - wu0, v - wv0
        if box is None:
            box_w = (int(uc) - 4, int(vc) - 4, int(uc) + 4, int(vc) + 4)
        else:
            box_w = (box[0] - wu0, box[1] - wv0, box[2] - wu0, box[3] - wv0)

        # 2) 标签 ROI 鲁棒深度
        z_roi, roi_sigma, roi_ratio, roi_n = robust_roi_depth(win, box_w, cfg)

        # 3) 面板平面
        ray = self.K.ray(u, v)
        plane, plane_msg = self._fit_panel_plane(win, wu0, wv0, uc, vc, z_roi)
        sigma_lat_px = cfg.pixel_sigma_px

        if plane is not None:
            p = intersect_ray_plane(ray, plane)
            cos_ang = abs(float(plane.normal @ ray)) / float(np.linalg.norm(ray))
            angle_ok = cos_ang >= math.cos(math.radians(cfg.max_ray_plane_angle_deg))
            if p is not None and angle_ok and cfg.min_depth_m < p[2] < cfg.max_depth_m:
                lever = float(np.linalg.norm(p - plane.centroid))
                n_in = max(plane.n_inliers, 1)
                tilt_sigma = plane.rms / (math.sqrt(n_in) * plane.spread)
                sigma_n = math.sqrt(plane.rms ** 2 / n_in + (lever * tilt_sigma) ** 2 + cfg.systematic_sigma_m ** 2)
                sigma_lat = p[2] * sigma_lat_px / self.K.fx
                msg = "面板平面定位"
                # 与标签 ROI 交叉校验：ROI 本身可靠且差异过大时，相信 ROI
                if z_roi is not None:
                    diff = abs(p[2] - z_roi)
                    if diff > cfg.max_plane_roi_disagreement_m:
                        if roi_ratio >= 0.5 and roi_sigma <= 0.002:
                            return self._roi_result(ray, z_roi, roi_sigma, roi_ratio, roi_n, plane,
                                                    "平面与标签深度相差 %.1f mm，改用标签 ROI" % (diff * 1000))
                        msg += "（与标签 ROI 相差 %.1f mm，ROI 不可靠，保留平面结果）" % (diff * 1000)
                return LocalizationResult(True, STATUS_PLANE, p, plane.normal, sigma_n, sigma_lat, roi_ratio,
                                          plane.inlier_ratio, plane.n_inliers, z_roi, msg)
            plane_msg = "射线与平面无有效交点或夹角过大"

        # 4) 后备：标签 ROI 中位数
        if z_roi is not None:
            return self._roi_result(ray, z_roi, roi_sigma, roi_ratio, roi_n, plane,
                                    "平面不可用（%s），改用标签 ROI" % plane_msg)
        return self._fail("标签 ROI 有效深度 %.0f%%，平面不可用（%s）" % (roi_ratio * 100, plane_msg), roi_ratio)

    # ------------------------------------------------------------------------------------------
    def _fit_panel_plane(self, win: np.ndarray, wu0: int, wv0: int, uc: float, vc: float,
                         z_roi: Optional[float]):
        cfg = self.cfg
        valid = (win > cfg.min_depth_m) & (win < cfg.max_depth_m)
        if valid.sum() < cfg.plane_min_points:
            return None, "窗口内有效深度点太少"
        # 参考深度：标签 ROI 优先，否则用窗口内中心附近有效深度的中位数
        if z_roi is not None:
            z_ref = z_roi
        else:
            vv, uu = np.nonzero(valid)
            dist2 = (uu - uc) ** 2 + (vv - vc) ** 2
            near = dist2 <= np.percentile(dist2, 30)
            z_ref = float(np.median(win[vv[near], uu[near]]))
        r_px = self.K.fx * cfg.plane_radius_m / z_ref
        vv, uu = np.nonzero(valid)
        z = win[vv, uu]
        sel = ((uu - uc) ** 2 + (vv - vc) ** 2 <= r_px ** 2) & (np.abs(z - z_ref) <= cfg.plane_depth_gate_m)
        uu, vv, z = uu[sel], vv[sel], z[sel]
        if uu.size < cfg.plane_min_points:
            return None, "平面候选点不足（%d）" % uu.size
        if uu.size > cfg.plane_max_points:
            pick = self.rng.choice(uu.size, cfg.plane_max_points, replace=False)
            uu, vv, z = uu[pick], vv[pick], z[pick]
        pts = deproject((uu + wu0).astype(np.float64), (vv + wv0).astype(np.float64), z.astype(np.float64), self.K)
        plane = fit_plane_ransac(pts, cfg.plane_inlier_thresh_m, cfg.plane_iters, self.rng)
        if plane is None:
            return None, "RANSAC 失败"
        if plane.n_inliers < cfg.plane_min_points or plane.inlier_ratio < cfg.plane_min_inlier_ratio:
            return None, "平面内点不足（%d 个，%.0f%%）" % (plane.n_inliers, plane.inlier_ratio * 100)
        if plane.spread_min < cfg.plane_min_extent_m:
            return None, "平面点集范围太小（%.1f mm）" % (plane.spread_min * 1000)
        return plane, ""

    def _roi_result(self, ray, z_roi, roi_sigma, roi_ratio, roi_n, plane, msg) -> LocalizationResult:
        p = ray * z_roi
        # 保守：用离散度本身，不除以 √n（主要误差来自边缘偏置），再叠加系统误差
        sigma_n = math.sqrt(max(roi_sigma, 1e-4) ** 2 + self.cfg.systematic_sigma_m ** 2)
        sigma_lat = z_roi * self.cfg.pixel_sigma_px / self.K.fx
        return LocalizationResult(True, STATUS_ROI_MEDIAN, p, None, sigma_n, sigma_lat, roi_ratio,
                                  plane.inlier_ratio if plane is not None else 0.0,
                                  plane.n_inliers if plane is not None else 0, z_roi, msg)

    @staticmethod
    def _fail(msg: str, roi_ratio: float) -> LocalizationResult:
        return LocalizationResult(False, STATUS_FAIL, None, None, float("nan"), float("nan"), roi_ratio,
                                  0.0, 0, None, msg)


def facing_yaw_from_normal(normal_base: np.ndarray) -> float:
    """
    面板法向（基座坐标系，指向机器人）-> 机械臂“面朝面板”的方向角（绕基座 z 轴，rad）。
    方向角 0 表示面朝基座 +x。
    """
    f = -np.asarray(normal_base, dtype=float)
    return math.atan2(f[1], f[0])
