"""第 1 步：纯仿真跑通全流程（不需要硬件）。

用法（在仓库根目录）：
    python3 -m robust_opt.simulate --out results/sim
    python3 -m robust_opt.simulate --out results/sim --quick     # 快速试跑

输出：
    fig1_success_vs_frames.png   成功率随帧数 n 的变化：基线 vs SAA 最优 dp
    fig2_saa_convergence.png     SAA 收敛性：样本内（乐观）vs 样本外成功率随 N 的变化
    fig3_distance.png            观测距离敏感性
    summary.md / summary.json    数值结果表

注意：噪声参数与容差均为假设值（见 noise_model.ParametricNoise / objective.Tolerance），
结论的"形状"可信，具体数字要等第 2 步换成实测数据。
"""
import argparse
import json
import os
import time

import numpy as np

from .evaluate import TABLE_HEAD, fmt_row, out_of_sample
from .noise_model import ParametricNoise
from .objective import Costs, Tolerance
from .plotting import C_BASE, C_SAA, C_SAA2, C_TEXT, L, plt
from .saa_solver import optimize_dp, scenario_sample_size, solve_model_a, solve_model_b

ZERO = np.zeros(3)


def exp_frames(noise, tol, costs, ns, N, N_eval, rng):
    rows = []
    for n in ns:
        m = noise.sample(N, n, rng)
        dp, _, _ = optimize_dp(m, tol, costs)
        base = out_of_sample(noise.sample, ZERO, n, tol, N_eval, rng)
        opt = out_of_sample(noise.sample, dp, n, tol, N_eval, rng)
        rows.append((n, base, opt))
    return rows


def exp_convergence(noise, tol, costs, n, Ns, R, N_eval, rng):
    out = []
    eval_m = noise.sample(N_eval, n, rng)        # 共用同一个大评估样本，减少评估噪声
    from .objective import rates
    for N in Ns:
        ins, oos = [], []
        for _ in range(R):
            m = noise.sample(N, n, rng)
            dp, s_in, _ = optimize_dp(m, tol, costs)
            ins.append(s_in)
            oos.append(rates(dp, eval_m, tol)[0])
        out.append((N, np.array(ins), np.array(oos)))
    return out


def exp_distance(noise, tol, costs, zs, N, N_eval, rng, n_robust=5):
    out = []
    for z in zs:
        nz = noise.with_(z=float(z))
        base = out_of_sample(nz.sample, ZERO, 1, tol, N_eval, rng)
        dp1, _, _ = optimize_dp(nz.sample(N, 1, rng), tol, costs)
        opt1 = out_of_sample(nz.sample, dp1, 1, tol, N_eval, rng)
        dpn, _, _ = optimize_dp(nz.sample(N, n_robust, rng), tol, costs)
        optn = out_of_sample(nz.sample, dpn, n_robust, tol, N_eval, rng)
        out.append((z, base, opt1, optn))
    return out


def _band(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)


def plot_frames(rows, path):
    ns = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, color, name in ((1, C_BASE, L("基线：dp = 0", "Baseline: dp = 0")),
                             (2, C_SAA, L("SAA 最优 dp", "SAA-optimal dp"))):
        y = [100 * r[key]["success"] for r in rows]
        lo = [100 * r[key]["success_ci"][0] for r in rows]
        hi = [100 * r[key]["success_ci"][1] for r in rows]
        _band(ax, ns, lo, hi, color)
        ax.plot(ns, y, "-o", color=color, label=name)
        ax.annotate(f"{y[-1]:.1f}%", (ns[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=C_TEXT, fontsize=9)
    ax.set_xlabel(L("观测帧数 n（≈ 耗时 s）", "Frames n (≈ seconds)"))
    ax.set_ylabel(L("样本外成功率 (%)", "Out-of-sample success (%)"))
    ax.set_title(L("多帧只能压低随机误差，系统偏差需要 dp 补偿", "Frames reduce random error; dp removes bias"),
                 color=C_TEXT, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(res, path):
    Ns = np.array([r[0] for r in res])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for idx, color, name in ((1, C_BASE, L("样本内（SAA 目标值）", "In-sample (SAA objective)")),
                             (2, C_SAA, L("样本外（真实性能）", "Out-of-sample (true)"))):
        med = np.array([100 * np.median(r[idx]) for r in res])
        lo = np.array([100 * np.percentile(r[idx], 10) for r in res])
        hi = np.array([100 * np.percentile(r[idx], 90) for r in res])
        _band(ax, Ns, lo, hi, color)
        ax.plot(Ns, med, "-o", color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel(L("SAA 样本量 N（对数轴）", "SAA sample size N (log)"))
    ax.set_ylabel(L("成功率 (%)，中位数与 10–90% 分位", "Success (%), median & 10–90%"))
    ax.set_title(L("SAA 收敛：小样本时样本内估计偏乐观", "SAA convergence: optimism bias shrinks with N"),
                 color=C_TEXT, fontsize=11)
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_distance(res, path, n_robust):
    zs = [100 * r[0] for r in res]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    series = ((1, C_BASE, L("基线：dp = 0，n = 1", "Baseline: dp = 0, n = 1")),
              (2, C_SAA, L("SAA dp，n = 1", "SAA dp, n = 1")),
              (3, C_SAA2, L(f"SAA dp，n = {n_robust}", f"SAA dp, n = {n_robust}")))
    for key, color, name in series:
        ax.plot(zs, [100 * r[key]["success"] for r in res], "-o", color=color, label=name)
    ax.set_xlabel(L("观测距离 z (cm)", "Viewing distance z (cm)"))
    ax.set_ylabel(L("样本外成功率 (%)", "Out-of-sample success (%)"))
    ax.set_title(L("距离越远，随机误差占比越大，多帧融合越有价值", "At distance, multi-frame fusion matters more"),
                 color=C_TEXT, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/sim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="缩小样本量快速试跑")
    ap.add_argument("--eps", type=float, default=0.10, help="模型 B 允许的失败概率")
    ap.add_argument("--lam", type=float, default=0.01, help="模型 A 每帧时间代价")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    noise, tol = ParametricNoise(), Tolerance()
    costs = Costs(lam=args.lam)
    q = args.quick
    N = 500 if q else 2000
    N_eval = 20000 if q else 200000
    ns = [1, 2, 3, 5, 8, 12]
    t0 = time.time()

    print("[1/4] 成功率 vs 帧数 ...")
    frames = exp_frames(noise, tol, costs, ns, N, N_eval, rng)
    plot_frames(frames, os.path.join(args.out, "fig1_success_vs_frames.png"))

    print("[2/4] 模型 A / 模型 B ...")
    best_a, rows_a = solve_model_a(noise.sample, ns, N, tol, costs, rng)
    best_b, feas_b, rows_b = solve_model_b(noise.sample, ns, N, tol, args.eps, rng)
    oos_a = out_of_sample(noise.sample, best_a.dp, best_a.n, tol, N_eval, rng)
    oos_b = out_of_sample(noise.sample, best_b.dp, best_b.n, tol, N_eval, rng)

    print("[3/4] SAA 收敛性 ...")
    Ns = [20, 50, 100, 200, 500, 1000] + ([] if q else [2000])
    conv = exp_convergence(noise, tol, costs, 3, Ns, 8 if q else 30, N_eval, rng)
    plot_convergence(conv, os.path.join(args.out, "fig2_saa_convergence.png"))

    print("[4/4] 观测距离敏感性 ...")
    zs = np.round(np.arange(0.25, 0.601, 0.05), 3)
    dist = exp_distance(noise, tol, costs, zs, N, N_eval, rng, n_robust=5)
    plot_distance(dist, os.path.join(args.out, "fig3_distance.png"), 5)

    nscen = scenario_sample_size(args.eps, 0.01, 3)
    lines = ["# 仿真结果（噪声参数与容差为假设值）", "",
             f"- 随机种子 {args.seed}；SAA 样本量 N = {N}；样本外评估 N' = {N_eval}",
             f"- 容差 (mm)：y ±{tol.y*1e3:.1f}，z ±{tol.z*1e3:.1f}，"
             f"x ∈ [-{tol.x_minus*1e3:.1f}, +{tol.x_plus*1e3:.1f}]",
             f"- 代价：c_fail = {costs.c_fail}，c_over = {costs.c_over}，lam = {costs.lam}/帧", "",
             "## 成功率 vs 帧数", "", TABLE_HEAD]
    for n, base, opt in frames:
        lines.append(fmt_row(f"基线 dp=0", base))
        lines.append(fmt_row(f"SAA 最优 dp", opt))
    lines += ["", "## 模型 A（期望代价）与模型 B（机会约束）", "", TABLE_HEAD,
              fmt_row(f"模型 A（lam={costs.lam}）", oos_a),
              fmt_row(f"模型 B（eps={args.eps}{'' if feas_b else '，不可行，取成功率最高'}）", oos_b), "",
              f"情景法样本量参考（eps={args.eps}, beta=0.01, d=3）：N ≥ {nscen}", "",
              "## SAA 收敛性（n = 3）", "", "| N | 样本内中位数 | 样本外中位数 | 样本外 10–90% |", "|---|---|---|---|"]
    for N_, ins, oos in conv:
        lines.append(f"| {N_} | {100*np.median(ins):.1f}% | {100*np.median(oos):.1f}% | "
                     f"{100*np.percentile(oos,10):.1f}–{100*np.percentile(oos,90):.1f}% |")
    lines += ["", "## 观测距离敏感性", "", "| z (cm) | 基线 n=1 | SAA n=1 | SAA n=5 |", "|---|---|---|---|"]
    for z, b, o1, o5 in dist:
        lines.append(f"| {100*z:.0f} | {100*b['success']:.1f}% | {100*o1['success']:.1f}% | {100*o5['success']:.1f}% |")
    lines += ["", f"耗时 {time.time()-t0:.1f} s"]
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"model_a": oos_a, "model_b": oos_b, "model_b_feasible": feas_b,
                   "frames": [{"n": n, "baseline": b, "saa": o} for n, b, o in frames],
                   "distance": [{"z": float(z), "baseline": b, "saa_n1": o1, "saa_n5": o5}
                                for z, b, o1, o5 in dist]}, f, ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"\n结果已写入 {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
