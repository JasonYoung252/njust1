"""样本外评估：用独立的大样本检验 SAA 解的真实性能（SAA 解在样本内往往偏乐观）。"""
import math

import numpy as np

from .objective import rates


def wilson_ci(p, n, z=1.96):
    """二项比例的 Wilson 95% 置信区间。"""
    if n == 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def out_of_sample(sampler, dp, n, tol, N_eval, rng):
    m = sampler(N_eval, n, rng)
    s, o = rates(dp, m, tol)
    lo, hi = wilson_ci(s, N_eval)
    return {"n": int(n), "dp_mm": [round(1000 * float(v), 3) for v in dp],
            "success": s, "success_ci": (lo, hi), "over": o}


def fmt_row(name, r):
    lo, hi = r["success_ci"]
    return (f"| {name} | {r['n']} | {r['dp_mm'][0]:+.2f}, {r['dp_mm'][1]:+.2f}, {r['dp_mm'][2]:+.2f} "
            f"| {100 * r['success']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] | {100 * r['over']:.1f}% |")


TABLE_HEAD = ("| 方案 | 帧数 n | dp (x, y, z) mm | 样本外成功率 [95% CI] | 过压率 |\n"
              "|---|---|---|---|---|")
