"""SAA（样本平均近似）求解器。

决策变量：补偿量 dp in R^3（连续）与观测帧数 n（整数）。
求解方式：对每个 n 抽 N 个样本，把期望/概率换成样本均值，在 dp 上做"粗网格 + 细网格"搜索。
指示函数不可导，且维数只有 3，网格搜索比梯度法更可靠，也能精确给出 SAA 问题的全局最优。

模型 A（期望代价）：
    min_{dp, n}  (1/N) sum_i [ c_fail 1{失败} + c_over 1{过压} ] + lam n
模型 B（机会约束，情景/SAA 版本）：
    min n   s.t.  (1/N) sum_i 1{成功} >= 1 - eps + gamma
"""
import math
from dataclasses import dataclass

import numpy as np

from .objective import Costs, expected_cost, grid_rates


@dataclass
class DpResult:
    n: int
    dp: np.ndarray
    success: float     # 样本内成功率
    over: float        # 样本内过压率
    cost: float        # 样本内期望代价（含 lam*n）


def _central_argmin(J, grids):
    """在所有最优网格点中取"最居中"的一个。

    指示函数型目标常有大片平台（很多 dp 样本内成功率都相同），
    直接 argmin 会偏向网格角落；取最优集合的质心再吸附到最近的最优点，解更稳健。
    """
    jmin = J.min()
    idx = np.argwhere(J <= jmin + 1e-12)
    pts = np.stack([grids[k][idx[:, k]] for k in range(3)], axis=1)
    centroid = pts.mean(axis=0)
    best = idx[np.argmin(((pts - centroid) ** 2).sum(axis=1))]
    return tuple(best)


def optimize_dp(m, tol, costs, half_range=0.006, steps=25, refine_steps=11, center=(0.0, 0.0, 0.0)):
    """固定 n、给定样本 m，求 SAA 内层问题的最优 dp。返回 (dp, 成功率, 过压率)。"""
    c = np.asarray(center, dtype=float)
    h, st = float(half_range), int(steps)
    result = None
    for _ in range(2):                      # 粗网格 -> 在最优点附近 ±1 个粗步长细化
        grids = [np.linspace(c[k] - h, c[k] + h, st) for k in range(3)]
        S, O = grid_rates(grids[0], grids[1], grids[2], m, tol)
        J = costs.c_fail * (1.0 - S) + costs.c_over * O[:, None, None]
        i, j, k = _central_argmin(J, grids)
        c = np.array([grids[0][i], grids[1][j], grids[2][k]])
        result = (c.copy(), float(S[i, j, k]), float(O[i]))
        h, st = 2.0 * h / (st - 1), int(refine_steps)
    return result


def solve_model_a(sampler, ns, N, tol, costs, rng, **grid_kw):
    """模型 A：对每个 n 求最优 dp，再在 n 上取期望代价最小者。返回 (最优, 每个 n 的结果列表)。"""
    rows = []
    for n in ns:
        m = sampler(N, n, rng)
        dp, s, o = optimize_dp(m, tol, costs, **grid_kw)
        rows.append(DpResult(n, dp, s, o, expected_cost(s, o, n, costs)))
    best = min(rows, key=lambda r: r.cost)
    return best, rows


def solve_model_b(sampler, ns, N, tol, eps, rng, gamma=0.0, **grid_kw):
    """模型 B：按 n 从小到大，找第一个满足 样本成功率 >= 1-eps+gamma 的方案。

    返回 (解, 是否可行, 每个 n 的结果列表)。若都不可行，返回成功率最高的方案并标记不可行。
    """
    succ_costs = Costs(c_fail=1.0, c_over=0.0, lam=0.0)   # 过压本身已计为失败
    rows = []
    for n in sorted(ns):
        m = sampler(N, n, rng)
        dp, s, o = optimize_dp(m, tol, succ_costs, **grid_kw)
        r = DpResult(n, dp, s, o, 1.0 - s)
        rows.append(r)
        if s >= 1.0 - eps + gamma:
            return r, True, rows
    return max(rows, key=lambda r: r.success), False, rows


def scenario_sample_size(eps, beta, d):
    """情景法（Calafiore-Campi）样本量充分条件：N >= (2/eps) (ln(1/beta) + d)。"""
    return int(math.ceil(2.0 / eps * (math.log(1.0 / beta) + d)))
