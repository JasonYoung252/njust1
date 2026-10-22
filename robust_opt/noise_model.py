"""随机变量模型：生成 n 帧聚合后的测量误差样本 m = p_hat - p_true（任务系，m）。

误差分为两类，这是建模的关键：
  * 随机分量（逐帧独立）：OCR 文本框中心抖动、深度噪声。多帧取中位数可以压低。
  * 系统分量（一次观测内固定）：深度偏差、手眼标定误差、机械臂绝对定位误差。
    多帧平均对它无效，只能靠补偿量 dp 或重新标定处理。

两种实现：
  ParametricNoise  —— 参数法，按相机投影模型传播噪声（仿真阶段用，参数为假设值）。
  EmpiricalNoise   —— 非参数法，直接对实测误差 bootstrap（拿到真机数据后用）。
二者都提供 sample(N, n, rng) -> (N, 3) 数组。
"""
from dataclasses import dataclass, field, replace

import numpy as np

# 相机系 -> 任务系的旋转（眼在手上、相机正对屏柜时的近似）：
#   相机 z（光轴，指向前方） -> 任务 x
#   相机 x（图像右）         -> 任务 -y
#   相机 y（图像下）         -> 任务 -z
R_TASK_CAM_DEFAULT = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])


def _aggregate(frames):
    """(N, n, 3) -> (N, 3)，按帧取中位数（与视觉端 n 帧融合方式一致）。"""
    if frames.shape[1] == 1:
        return frames[:, 0, :]
    return np.median(frames, axis=1)


@dataclass
class ParametricNoise:
    """参数法噪声模型。所有默认值都是假设值，拿到实测数据后应替换。"""
    z: float = 0.35              # 观测距离（相机到标签），m
    fx: float = 1380.0           # 彩色相机焦距（1920x1080 时约 1380 px）
    fy: float = 1380.0
    du0: float = 0.0             # 标签中心相对主点的像素偏移
    dv0: float = 0.0
    sigma_uv: float = 1.0        # OCR 文本框中心抖动，px（随机）
    depth_sigma0: float = 0.0005  # 深度噪声 sigma_z = s0 + s2 * z^2，m（随机）
    depth_sigma2: float = 0.005
    depth_bias: float = 0.0015   # 深度系统偏差，m（系统，沿光轴，正值表示测得偏远）
    he_rot_sigma: float = 0.002  # 手眼旋转误差，rad，每轴（系统）
    he_trans_sigma: float = 0.0008  # 手眼平移误差，m，每轴（系统）
    arm_sigma: float = 0.0003    # 机械臂绝对定位误差，m，每轴（系统）
    # 任务系下的固定偏差（如 OCR 文本框中心与标签物理中心不重合），m。
    # 代码里 offset[1]-0.002 这类手调常数，对应的就是这种偏差。
    task_bias: tuple = (0.0, 0.0015, 0.0)
    R_task_cam: np.ndarray = field(default_factory=lambda: R_TASK_CAM_DEFAULT.copy())

    def sigma_z(self):
        return self.depth_sigma0 + self.depth_sigma2 * self.z ** 2

    def with_(self, **kw):
        return replace(self, **kw)

    def sample(self, N, n, rng):
        z = self.z
        xn, yn = self.du0 / self.fx, self.dv0 / self.fy      # 归一化像平面坐标
        p_c = np.array([z * xn, z * yn, z])                  # 名义相机系坐标

        # ---- 随机分量：逐帧独立 ----
        du = rng.normal(0.0, self.sigma_uv, (N, n))
        dv = rng.normal(0.0, self.sigma_uv, (N, n))
        dz = rng.normal(0.0, self.sigma_z(), (N, n))
        frames = np.empty((N, n, 3))
        frames[..., 0] = dz * xn + (z + dz) * du / self.fx  # 由 X = Z*(u-cx)/fx 线性化
        frames[..., 1] = dz * yn + (z + dz) * dv / self.fy
        frames[..., 2] = dz
        m_cam = _aggregate(frames)

        # ---- 系统分量：一次观测内固定 ----
        m_cam = m_cam + self.depth_bias * np.array([xn, yn, 1.0])
        w = rng.normal(0.0, self.he_rot_sigma, (N, 3))       # 小角度旋转向量
        m_cam = m_cam + np.cross(w, p_c)
        m_cam = m_cam + rng.normal(0.0, self.he_trans_sigma, (N, 3))

        m_task = m_cam @ self.R_task_cam.T
        m_task = m_task + np.asarray(self.task_bias, dtype=float)
        m_task = m_task + rng.normal(0.0, self.arm_sigma, (N, 3))
        return m_task


class EmpiricalNoise:
    """非参数（bootstrap）噪声模型。

    systematic : (K, 3) 真值实验得到的每次观测误差（任务系）。每次观测已用多帧中位数，
                 残留的随机部分很小，近似视为系统误差。
    jitter     : (J, 3) 静态重复测量的逐帧残差（逐帧值减去该组中位数，任务系）。
    采样：随机抽一个系统误差 + 有放回抽 n 帧残差取中位数。
    """

    def __init__(self, systematic, jitter):
        self.systematic = np.asarray(systematic, dtype=float).reshape(-1, 3)
        self.jitter = np.asarray(jitter, dtype=float).reshape(-1, 3)
        if len(self.systematic) == 0:
            raise ValueError("systematic 为空：需要真值实验数据（collect_gt.py）")
        if len(self.jitter) == 0:
            raise ValueError("jitter 为空：需要静态重复测量数据（n_frames 较大的视觉日志）")

    def sample(self, N, n, rng):
        s = self.systematic[rng.integers(len(self.systematic), size=N)]
        j = self.jitter[rng.integers(len(self.jitter), size=(N, n))]
        return s + _aggregate(j)
