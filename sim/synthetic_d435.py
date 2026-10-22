#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合成 D435 对齐深度图：屏柜面板 + 标签 + 一排凸出的压板，带有 D435 常见的失效模式。

用途：没有真机时，用已知真值比较“原程序取深度的方式”和 depth_localizer 的定位精度与失败率。
注意：这是简化模型，用来做方法对比和代码验证；论文里的最终数字必须来自真机实验。

模拟的失效模式（参数见 SceneParams）：
    - 深度噪声：σ = z² · subpixel / (f_depth · baseline)，按 Intel 调参文档的 RMS 公式
    - 反光压板正面：大面积、成片的空洞（多帧之间大部分位置不变）
    - 光滑标签：成片空洞
    - 面板：少量成片空洞（深色、反光区域）+ 每帧随机散点空洞
    - 反光区域的错误深度：少量像素给出偏差 1–3 cm 的错误值（不是 0）
    - 立体遮挡带：凸出物体左侧一条无效带，宽度 ≈ f · B · (1/z_前 − 1/z_后)
    - 深度到彩色对齐的遮挡带：凸出物体右侧一条较窄的无效带（RGB 与左红外相机的基线）

坐标：相机 x 右、y 下、z 前（m）。面板坐标：x_P 右、y_P 上、z_P 指向相机，原点在标签中心。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SceneParams:
    # 相机（模拟对齐到 1920×1080 彩色图后再降采样一半，即 960×540）
    width: int = 960
    height: int = 540
    fx: float = 698.8
    fy: float = 698.8
    cx: float = 480.0
    cy: float = 270.0
    # D435 深度相机参数（用于噪声和遮挡带）
    depth_fx: float = 447.0          # 848×480、约 87° 水平视场
    baseline_m: float = 0.050        # 左右红外相机基线
    rgb_baseline_m: float = 0.015    # 左红外到 RGB 相机的距离
    subpixel: float = 0.08
    # 几何
    distance_m: float = 0.35         # 相机到标签的距离（沿光轴）
    yaw_deg: float = 0.0             # 面板相对相机的偏航（绕竖直轴）
    pitch_deg: float = 0.0
    label_offset_px: Tuple[float, float] = (0.0, 0.0)   # 标签在图像中相对主点的偏移
    label_size_m: Tuple[float, float] = (0.030, 0.010)
    plate_size_m: Tuple[float, float] = (0.014, 0.040)
    plate_protrusion_m: float = 0.018
    plate_row_y_m: float = 0.035                         # 压板中心在标签上方的高度
    plate_xs_m: Tuple[float, ...] = (-0.06, -0.03, 0.0, 0.03, 0.06)
    # 失效
    label_dropout: float = 0.3       # 标签区域空洞比例
    plate_dropout: float = 0.9       # 压板正面空洞比例
    panel_dropout: float = 0.05      # 面板成片空洞比例
    speckle: float = 0.03            # 每帧随机散点空洞
    persistence: float = 0.85        # 成片空洞在多帧之间保持不变的比例
    glossy_outlier: float = 0.10     # 反光区域（标签、压板）中，未成为空洞的像素里出现错误深度的比例
    glossy_outlier_sigma_m: float = 0.015  # 错误深度的偏差尺度
    blob_scale_px: int = 6           # 成片空洞的尺度


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class SyntheticScene:
    def __init__(self, params: SceneParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng
        p = params
        R = _rot_y(np.radians(p.yaw_deg)).dot(_rot_x(np.radians(p.pitch_deg)))
        self.x_axis = R[:, 0]                 # 面板 x_P（向右）
        self.y_axis = -R[:, 1]                # 面板 y_P（向上，相机 y 向下）
        self.normal = -R[:, 2]                # 面板法向，指向相机
        u0 = p.cx + p.label_offset_px[0]
        v0 = p.cy + p.label_offset_px[1]
        ray = np.array([(u0 - p.cx) / p.fx, (v0 - p.cy) / p.fy, 1.0])
        self.label_center = ray * p.distance_m           # 标签中心（真值，相机坐标）
        self.label_uv = (u0, v0)

        H, W = p.height, p.width
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        rays = np.stack([(uu - p.cx) / p.fx, (vv - p.cy) / p.fy, np.ones_like(uu)], axis=-1)
        self.uu, self.vv = uu, vv

        # 面板平面求交
        denom = rays.dot(self.normal)
        t_panel = (self.normal.dot(self.label_center)) / denom
        depth = t_panel.copy()
        P = rays * t_panel[..., None] - self.label_center
        xP, yP = P.dot(self.x_axis), P.dot(self.y_axis)
        lw, lh = p.label_size_m
        self.label_mask = (np.abs(xP) <= lw / 2) & (np.abs(yP) <= lh / 2)

        # 压板正面求交（只建模正面，侧面很窄）
        front_point = self.label_center + self.normal * p.plate_protrusion_m
        t_front = (self.normal.dot(front_point)) / denom
        Pf = rays * t_front[..., None] - self.label_center
        xF, yF = Pf.dot(self.x_axis), Pf.dot(self.y_axis)
        pw, ph = p.plate_size_m
        plate = np.zeros((H, W), dtype=bool)
        for x0 in p.plate_xs_m:
            plate |= (np.abs(xF - x0) <= pw / 2) & (np.abs(yF - p.plate_row_y_m) <= ph / 2)
        depth = np.where(plate, t_front, depth)
        self.plate_mask = plate
        self.true_depth = depth                       # 真值深度（rays 的 z = 1，t 即深度）

        # 遮挡带：凸出物体左侧（立体匹配）和右侧（对齐到彩色）
        z = p.distance_m
        w_stereo = int(round(p.fx * p.baseline_m * (1.0 / (z - p.plate_protrusion_m) - 1.0 / z)))
        w_align = int(round(p.fx * p.rgb_baseline_m * (1.0 / (z - p.plate_protrusion_m) - 1.0 / z)))
        shadow = np.zeros_like(plate)
        for k in range(1, max(w_stereo, 0) + 1):
            shadow[:, :-k] |= plate[:, k:]
        for k in range(1, max(w_align, 0) + 1):
            shadow[:, k:] |= plate[:, :-k]
        self.shadow_mask = shadow & ~plate

        # 成片空洞的“持久”部分（多帧之间不变）
        self.persistent_holes = self._region_holes(persistent=True)

    # ------------------------------------------------------------------------------------------
    def _blob_field(self) -> np.ndarray:
        """低分辨率随机场放大，得到成片的随机值，用于生成成片空洞。"""
        p = self.p
        s = max(1, p.blob_scale_px)
        h, w = p.height // s + 2, p.width // s + 2
        small = self.rng.random((h, w))
        big = np.kron(small, np.ones((s, s)))[:p.height, :p.width]
        return big

    def _holes_with_ratio(self, mask: np.ndarray, ratio: float) -> np.ndarray:
        if ratio <= 0 or not mask.any():
            return np.zeros_like(mask)
        if ratio >= 1:
            return mask.copy()
        field = self._blob_field()
        thr = np.quantile(field[mask], ratio)
        return mask & (field <= thr)

    def _region_holes(self, persistent: bool) -> np.ndarray:
        """
        persistent=True：多帧不变的成片空洞，占 dropout × persistence。
        persistent=False：每帧变化的部分，从其余像素中选取，使每帧的空洞比例约等于 dropout。
        """
        p = self.p
        panel = ~self.plate_mask & ~self.label_mask
        holes = np.zeros_like(self.plate_mask)
        for mask, ratio in ((self.plate_mask, p.plate_dropout), (self.label_mask, p.label_dropout),
                            (panel, p.panel_dropout)):
            base = ratio * p.persistence
            if persistent:
                holes |= self._holes_with_ratio(mask, base)
            else:
                rest = mask & ~self.persistent_holes
                extra = (ratio - base) / max(1.0 - base, 1e-6)
                holes |= self._holes_with_ratio(rest, extra)
        return holes

    def render_frames(self, n_frames: int) -> np.ndarray:
        """返回 (N, H, W) 的深度（m），0 为无效。"""
        p = self.p
        frames = []
        sigma = self.true_depth ** 2 * p.subpixel / (p.depth_fx * p.baseline_m)
        for _ in range(n_frames):
            d = self.true_depth + self.rng.normal(0.0, 1.0, self.true_depth.shape) * sigma
            invalid = self.persistent_holes | self._region_holes(persistent=False) | self.shadow_mask
            invalid |= self.rng.random(d.shape) < p.speckle
            # 镜面反射区域偶尔匹配错，得到错误但非零的深度
            glossy = (self.label_mask | self.plate_mask) & (self.rng.random(d.shape) < p.glossy_outlier)
            d = np.where(glossy, d + self.rng.normal(0.0, p.glossy_outlier_sigma_m, d.shape), d)
            frames.append(np.where(invalid, 0.0, d).astype(np.float32))
        return np.stack(frames, axis=0)

    def label_polygon(self) -> List[Tuple[float, float]]:
        """标签四个角在图像中的投影（相当于理想的 OCR 文字框）。"""
        p = self.p
        lw, lh = p.label_size_m
        pts = []
        for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1)):
            X = self.label_center + self.x_axis * sx * lw / 2 + self.y_axis * sy * lh / 2
            pts.append((p.fx * X[0] / X[2] + p.cx, p.fy * X[1] / X[2] + p.cy))
        return pts

    def panel_to_camera(self, offset_panel: np.ndarray, axes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                        origin: Optional[np.ndarray] = None) -> np.ndarray:
        """面板坐标系下的偏移 -> 相机坐标。axes 为 (x_P, y_P, n)，默认用真值。"""
        x_ax, y_ax, n_ax = axes if axes is not None else (self.x_axis, self.y_axis, self.normal)
        o = origin if origin is not None else self.label_center
        return o + x_ax * offset_panel[0] + y_ax * offset_panel[1] + n_ax * offset_panel[2]
