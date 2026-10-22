"""第 3 步：用真机数据求解，并生成可直接给 arm_control 使用的补偿配置。

用法（在仓库根目录）：
    python3 -m robust_opt.fit_from_data \\
        --vision "~/robust_opt_logs/vision_*.csv" --gt ~/robust_opt_logs/gt.csv \\
        --out-yaml config/compensation_robust.yaml

之后用  roslaunch njust rm_63_ts.launch comp_profile:=robust  切换到优化方案，
comp_profile:=baseline 回到原来的手调常数，二者做真机对比。

真值轮数 >= 10 时自动按 --holdout 比例留出测试集，报告的是测试集（样本外）结果；
少于 10 轮时只能报告样本内结果，会给出提示。
"""
import argparse
import os
import time

import numpy as np

from .data import load_jitter, load_systematic
from .evaluate import TABLE_HEAD, fmt_row, out_of_sample
from .noise_model import EmpiricalNoise
from .objective import Costs, Tolerance
from .saa_solver import solve_model_a, solve_model_b

# 当前 arm_control.py 中的手调常数（任务系 m），即真机对比的基线
HAND_CONSTANTS = {
    "switch_open_approach": [0.0, -0.002, 0.0],
    "switch_open_return": [0.0, -0.0022, 0.0018],
    "switch_close_approach": [0.0, 0.0, 0.0025],
}


def _mm(v):
    return ", ".join(f"{1000 * x:+.2f}" for x in v)


def write_yaml(path, dp, n, note_lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    vec = "[" + ", ".join(f"{x:.5f}" for x in dp) + "]"
    lines = ["# 由 robust_opt.fit_from_data 自动生成，" + time.strftime("%Y-%m-%d %H:%M"),
             "# 任务坐标系：面朝标签，前 x、左 y、上 z（与 arm_control.dxy 一致），单位 m"]
    lines += ["# " + s for s in note_lines]
    lines += ["compensation:",
              f"  switch_open_approach: {vec}",
              f"  switch_open_return: {vec}",
              f"  switch_close_approach: {vec}",
              f"n_frames: {int(n)}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vision", nargs="+", required=True, help="vision_*.csv，可用通配符")
    ap.add_argument("--gt", nargs="+", required=True, help="gt.csv")
    ap.add_argument("--out-yaml", default="config/compensation_robust.yaml")
    ap.add_argument("--min-frames", type=int, default=10, help="静态测量组的最少帧数")
    ap.add_argument("--default-direction", type=int, default=None, help="日志中方向未知时使用的方向")
    ap.add_argument("--tol", type=float, nargs=4, metavar=("Y", "Z", "X_MINUS", "X_PLUS"),
                    default=[2.5, 2.5, 3.0, 1.5], help="容差，mm")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--N-eval", type=int, default=100000)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    expand = lambda ps: [os.path.expanduser(p) for p in ps]
    jitter, n_groups = load_jitter(expand(args.vision), args.min_frames, args.default_direction)
    systematic, dist = load_systematic(expand(args.gt))
    print(f"静态测量组 {n_groups} 组 / 逐帧残差 {len(jitter)} 个；真值实验 {len(systematic)} 轮")
    if len(jitter) == 0 or len(systematic) == 0:
        raise SystemExit("数据不足：需要至少一组 n_frames>=min_frames 的静态测量，以及 gt.csv 中至少一轮真值")
    print(f"逐帧随机误差 std (mm)：{_mm(jitter.std(axis=0))}")
    print(f"系统误差 均值 (mm)：{_mm(systematic.mean(axis=0))}；std (mm)：{_mm(systematic.std(axis=0))}")

    rng = np.random.default_rng(args.seed)
    tol = Tolerance(y=args.tol[0] / 1000, z=args.tol[1] / 1000,
                    x_minus=args.tol[2] / 1000, x_plus=args.tol[3] / 1000)
    costs = Costs(lam=args.lam)

    K = len(systematic)
    if K >= 10:
        idx = rng.permutation(K)
        k_test = max(3, int(round(args.holdout * K)))
        test, train = systematic[idx[:k_test]], systematic[idx[k_test:]]
        scope = f"样本外（留出 {k_test}/{K} 轮真值作测试集）"
    else:
        train = test = systematic
        scope = f"样本内（真值仅 {K} 轮，建议 >= 10 轮以做留出验证）"
    train_s, test_s = EmpiricalNoise(train, jitter), EmpiricalNoise(test, jitter)

    best_a, _ = solve_model_a(train_s.sample, args.ns, args.N, tol, costs, rng)
    best_b, feas_b, _ = solve_model_b(train_s.sample, args.ns, args.N, tol, args.eps, rng)

    ev = lambda dp, n: out_of_sample(test_s.sample, np.asarray(dp, float), n, tol, args.N_eval, rng)
    lines = [f"评估口径：{scope}", "", TABLE_HEAD,
             fmt_row("不补偿 dp=0", ev(np.zeros(3), 1))]
    for k, v in HAND_CONSTANTS.items():
        lines.append(fmt_row(f"手调常数 {k}", ev(v, 1)))
    lines.append(fmt_row(f"模型 A（lam={args.lam}）", ev(best_a.dp, best_a.n)))
    lines.append(fmt_row(f"模型 B（eps={args.eps}{'' if feas_b else '，不可行'}）", ev(best_b.dp, best_b.n)))
    print("\n".join(lines))

    chosen, name = (best_b, "模型 B") if feas_b else (best_a, "模型 A")
    write_yaml(args.out_yaml, chosen.dp, chosen.n, [
        f"采用 {name}：dp (mm) = {_mm(chosen.dp)}，n_frames = {chosen.n}",
        f"容差 mm (y, z, x-, x+) = {tuple(args.tol)}；真值 {K} 轮；静态组 {n_groups} 组",
        "三个压板动作使用同一个视觉补偿量；如需区分动作，可手动修改"])
    print(f"\n已写入 {os.path.abspath(args.out_yaml)}（{name}）")


if __name__ == "__main__":
    main()
