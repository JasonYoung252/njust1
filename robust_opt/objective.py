"""成功判定与代价函数。

执行误差 e = dp + m（任务系）。成功条件
    |e_y| <= tau_y,  |e_z| <= tau_z,  -tau_x_minus <= e_x <= tau_x_plus
e_x > tau_x_plus 视为"过压/碰撞"（比普通失败更严重，额外计 c_over）。
法向容差不对称（tau_x_minus != tau_x_plus）是 dp 不为零的根本原因之一。

代价（模型 A）：
    L(dp, n; m) = c_fail * 1{失败} + c_over * 1{过压} + lam * n
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Tolerance:
    """容差，单位 m。默认值为假设值，需按压板与夹爪实际尺寸修改。"""
    y: float = 0.0025
    z: float = 0.0025
    x_minus: float = 0.0030   # 压不到位方向
    x_plus: float = 0.0015    # 压过头方向（更危险，通常更小）


@dataclass
class Costs:
    c_fail: float = 1.0       # 失败代价
    c_over: float = 2.0       # 过压额外代价（叠加在失败之上）
    lam: float = 0.0          # 每多一帧的时间代价
    t_frame: float = 1.0      # 每帧耗时，s（PaddleOCR CPU 约 1 s/帧，用于画图）


def axis_masks(grid_x, grid_y, grid_z, m, tol):
    """对乘积网格逐轴计算成功指示，利用可分离性避免 G*N*3 的大数组。

    返回 Ax (gx, N), Ay (gy, N), Az (gz, N), Ox (gx, N)，均为 float32 的 0/1。
    """
    ex = grid_x[:, None] + m[None, :, 0]
    ey = grid_y[:, None] + m[None, :, 1]
    ez = grid_z[:, None] + m[None, :, 2]
    over = ex > tol.x_plus
    ax = (ex >= -tol.x_minus) & ~over
    ay = np.abs(ey) <= tol.y
    az = np.abs(ez) <= tol.z
    f = np.float32
    return ax.astype(f), ay.astype(f), az.astype(f), over.astype(f)


def grid_rates(grid_x, grid_y, grid_z, m, tol):
    """网格上每个 dp 的样本成功率 S (gx, gy, gz) 与过压率 O (gx,)。"""
    ax, ay, az, ox = axis_masks(grid_x, grid_y, grid_z, m, tol)
    N = m.shape[0]
    S = np.einsum("in,jn,kn->ijk", ax, ay, az, optimize=True) / N
    O = ox.mean(axis=1)
    return S.astype(float), O.astype(float)


def rates(dp, m, tol):
    """单个 dp 的成功率与过压率（用于样本外评估）。"""
    e = np.asarray(dp, dtype=float)[None, :] + m
    over = e[:, 0] > tol.x_plus
    ok = (e[:, 0] >= -tol.x_minus) & ~over & (np.abs(e[:, 1]) <= tol.y) & (np.abs(e[:, 2]) <= tol.z)
    return float(ok.mean()), float(over.mean())


def expected_cost(success, over, n, costs):
    return costs.c_fail * (1.0 - success) + costs.c_over * over + costs.lam * n
