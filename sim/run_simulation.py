#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒙特卡洛仿真：比较原程序的取深度方式与 depth_localizer 的定位效果。

    python3 sim/run_simulation.py            # 默认 80 次/条件，约 1–2 分钟
    python3 sim/run_simulation.py --trials 30 --quick

输出（sim/results/）：
    summary.json / summary.md     各方法的成功率、误差统计
    fig_success_vs_holes.png      成功率随标签空洞比例的变化
    fig_error_vs_yaw.png          操作点误差随面板偏航的变化
    fig_example_frame.png         一帧合成深度图示例

对比的方法：
    old_pixel   原程序：OCR 中心单像素深度（为 0 时得到原点，由 arm_control 的检查判为无效后重试）
    old_circle  原程序中已写好但未使用的圆形区域平均深度（半径 5 像素，忽略 0）
    roi_median  新模块的后备方法：多帧融合 + 标签框内中位数
    plane       新模块的主方法：面板平面 RANSAC + 射线求交
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)

from depth_localizer import CameraIntrinsics, DepthLocalizer, LocalizerConfig, box_from_polygon  # noqa: E402
from synthetic_d435 import SceneParams, SyntheticScene  # noqa: E402

METHODS = ["old_pixel", "old_circle", "roi_median", "plane"]
METHOD_LABELS = {
    "old_pixel": "原程序：单像素",
    "old_circle": "原程序：圆形平均（未启用）",
    "roi_median": "新：多帧 ROI 中位数",
    "plane": "新：面板平面 + 射线求交",
}
# 操作点：标签右侧 3 cm、上方 3.5 cm 的压板正面中心（面板坐标）
OPERATING_OFFSET = np.array([0.03, 0.035, 0.018])
SILENT_ERROR_MM = 5.0


def nominal_axes():
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])


def axes_from_normal(n):
    """由估计的面板法向构造面板坐标轴（y 取相机“上”方向在面板内的投影）。"""
    n = n / np.linalg.norm(n)
    up = np.array([0.0, -1.0, 0.0])
    y = up - n * up.dot(n)
    y /= np.linalg.norm(y)
    x = np.cross(y, n)
    return x, y, n


def run_trial(params, rng, n_frames, pixel_jitter=1.5):
    scene = SyntheticScene(params, rng)
    frames = scene.render_frames(n_frames)
    K = CameraIntrinsics(params.fx, params.fy, params.cx, params.cy)
    poly = scene.label_polygon()
    center = np.mean(np.array(poly), axis=0) + rng.normal(0, pixel_jitter, 2)
    u, v = int(center[0]), int(center[1])   # 原程序取整方式
    truth = scene.label_center
    op_truth = scene.panel_to_camera(OPERATING_OFFSET)
    out = {}

    # 原程序：单像素（OCR 用第 0 帧，取深度时重新抓帧 -> 用第 1 帧）
    f_old = frames[min(1, n_frames - 1)]
    z = float(f_old[v, u])
    ray = K.ray(u, v)
    out["old_pixel"] = _record(ray * z if z > 0 else None, None, truth, op_truth)

    # 原程序中未启用的圆形平均
    r = 5
    yy, xx = np.mgrid[v - r:v + r + 1, u - r:u + r + 1]
    patch = f_old[v - r:v + r + 1, u - r:u + r + 1]
    m = ((xx - u) ** 2 + (yy - v) ** 2 <= r * r) & (patch > 0)
    z = float(patch[m].mean()) if m.any() else 0.0
    out["old_circle"] = _record(ray * z if z > 0 else None, None, truth, op_truth)

    box = box_from_polygon(poly)
    t0 = time.perf_counter()
    res_roi = DepthLocalizer(K, LocalizerConfig(plane_min_points=10 ** 9)).localize(frames, (u, v), box)
    out["roi_median"] = _record(res_roi.point_cam if res_roi.ok else None, None, truth, op_truth)
    t1 = time.perf_counter()
    res = DepthLocalizer(K, LocalizerConfig()).localize(frames, (u, v), box)
    t2 = time.perf_counter()
    out["plane"] = _record(res.point_cam if res.ok else None, res.normal_cam, truth, op_truth,
                           sigma=(res.sigma_normal_m, res.sigma_lateral_m), status=res.status)
    out["plane"]["time_ms"] = (t2 - t1) * 1000
    out["roi_median"]["time_ms"] = (t1 - t0) * 1000
    out["_true_normal"] = scene.normal
    return out


def _record(point, normal, truth, op_truth, sigma=None, status=None):
    rec = {"ok": point is not None, "status": status}
    if point is None:
        return rec
    rec["err_mm"] = float(np.linalg.norm(point - truth) * 1000)
    rec["err_vec"] = (point - truth) * 1000
    op_nom = point + sum(a * o for a, o in zip(nominal_axes(), OPERATING_OFFSET))
    rec["op_err_nominal_mm"] = float(np.linalg.norm(op_nom - op_truth) * 1000)
    if normal is not None:
        ax = axes_from_normal(normal)
        op_est = point + sum(a * o for a, o in zip(ax, OPERATING_OFFSET))
        rec["op_err_corrected_mm"] = float(np.linalg.norm(op_est - op_truth) * 1000)
        rec["normal"] = normal
    if sigma is not None:
        rec["sigma_normal_mm"] = sigma[0] * 1000
        rec["sigma_lateral_mm"] = sigma[1] * 1000
    return rec


def random_params(rng, **override):
    p = SceneParams(
        distance_m=float(rng.uniform(0.30, 0.40)),
        yaw_deg=float(rng.uniform(-8, 8)),
        pitch_deg=float(rng.uniform(-3, 3)),
        label_offset_px=(float(rng.uniform(-150, 150)), float(rng.uniform(-60, 80))),
        label_dropout=float(rng.uniform(0.0, 0.9)),
        plate_dropout=float(rng.uniform(0.7, 0.98)),
        panel_dropout=float(rng.uniform(0.0, 0.15)),
        speckle=float(rng.uniform(0.01, 0.05)),
    )
    for k, v in override.items():
        setattr(p, k, v)
    return p


def summarize(records):
    s = {}
    for m in METHODS:
        rs = [r[m] for r in records]
        ok = [r for r in rs if r["ok"]]
        errs = np.array([r["err_mm"] for r in ok]) if ok else np.array([])
        d = {
            "n": len(rs),
            "success_rate": len(ok) / float(len(rs)),
            "median_err_mm": float(np.median(errs)) if errs.size else None,
            "p95_err_mm": float(np.percentile(errs, 95)) if errs.size else None,
            "silent_wrong_rate": float(np.mean(errs > SILENT_ERROR_MM) * len(ok) / len(rs)) if errs.size else 0.0,
        }
        times = [r.get("time_ms") for r in ok if r.get("time_ms") is not None]
        if times:
            d["median_time_ms"] = float(np.median(times))
        if m == "plane":
            st = [r["status"] for r in rs]
            d["share_plane"] = st.count(1) / float(len(rs))
            d["share_roi_fallback"] = st.count(2) / float(len(rs))
            sig = np.array([r["sigma_normal_mm"] for r in ok if "sigma_normal_mm" in r])
            ez = np.array([abs(r["err_vec"].dot(np.array([0, 0, 1.0]))) for r in ok])
            if sig.size:
                d["median_sigma_normal_mm"] = float(np.median(sig))
                d["coverage_3sigma_depth"] = float(np.mean(ez <= 3 * np.maximum(sig, 0.3)))
        s[m] = d
    return s


def experiment_holes(rng, trials, n_frames):
    levels = np.round(np.linspace(0.0, 1.0, 11), 2)
    table = {m: [] for m in METHODS}
    errs = {m: [] for m in METHODS}
    for lv in levels:
        recs = [run_trial(random_params(rng, label_dropout=float(lv)), rng, n_frames) for _ in range(trials)]
        s = summarize(recs)
        for m in METHODS:
            table[m].append(s[m]["success_rate"])
            errs[m].append(s[m]["median_err_mm"])
    return levels, table, errs


def experiment_yaw(rng, trials, n_frames):
    yaws = np.arange(0, 13, 2)
    res = {"old_pixel_nominal": [], "plane_nominal": [], "plane_corrected": []}
    for yw in yaws:
        a, b, c = [], [], []
        for _ in range(trials):
            sign = 1 if rng.random() < 0.5 else -1
            p = random_params(rng, yaw_deg=float(sign * yw), pitch_deg=0.0, label_dropout=float(rng.uniform(0, 0.6)))
            r = run_trial(p, rng, n_frames)
            if r["old_pixel"]["ok"]:
                a.append(r["old_pixel"]["op_err_nominal_mm"])
            if r["plane"]["ok"] and "op_err_corrected_mm" in r["plane"]:
                b.append(r["plane"]["op_err_nominal_mm"])
                c.append(r["plane"]["op_err_corrected_mm"])
        for key, arr in (("old_pixel_nominal", a), ("plane_nominal", b), ("plane_corrected", c)):
            arr = np.array(arr)
            res[key].append([float(np.percentile(arr, 25)), float(np.median(arr)), float(np.percentile(arr, 75))]
                            if arr.size else [np.nan] * 3)
    return yaws, res


# ----------------------------------------------------------------------------------------------
# 作图
# ----------------------------------------------------------------------------------------------
COLORS = {"old_pixel": "#8a8984", "old_circle": "#eb6834", "roi_median": "#1baf7a", "plane": "#2a78d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK TC", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 11,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    })
    return plt


def plot_holes(levels, table, path):
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    x = levels * 100
    for m in METHODS:
        y = np.array(table[m]) * 100
        ax.plot(x, y, color=COLORS[m], lw=2, marker="o", ms=5, label=METHOD_LABELS[m])
    ax.set_xlabel("标签区域深度空洞比例（%）")
    ax.set_ylabel("单次定位成功率（%）")
    ax.set_ylim(-3, 103)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("标签全是空洞时，只有面板平面方法仍能定位", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.18))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_yaw(yaws, res, path):
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    spec = [("old_pixel_nominal", "原程序（单像素 + 固定方向偏移）", COLORS["old_pixel"]),
            ("plane_nominal", "新定位 + 固定方向偏移", COLORS["roi_median"]),
            ("plane_corrected", "新定位 + 按实测法向修正偏移", COLORS["plane"])]
    for key, label, c in spec:
        arr = np.array(res[key])
        ax.fill_between(yaws, arr[:, 0], arr[:, 2], color=c, alpha=0.12, lw=0)
        ax.plot(yaws, arr[:, 1], color=c, lw=2, marker="o", ms=5, label=label)
    ax.set_xlabel("面板相对名义方向的偏航（°）")
    ax.set_ylabel("压板操作点误差（mm，中位数与四分位区间）")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("偏航越大，固定方向偏移的误差越大；按法向修正后基本不随偏航增长", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_example(rng, path):
    plt = _setup_mpl()
    p = SceneParams(distance_m=0.35, yaw_deg=5.0, label_dropout=0.6, plate_dropout=0.92, panel_dropout=0.08)
    scene = SyntheticScene(p, rng)
    frame = scene.render_frames(1)[0]
    u, v = scene.label_uv
    r = 170
    crop = frame[int(v) - r // 2 - 40:int(v) + r // 2 + 10, int(u) - r:int(u) + r]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    disp = np.ma.masked_where(crop <= 0, crop * 1000)
    cmap = plt.get_cmap("Blues_r").copy()
    cmap.set_bad("#0b0b0b")
    im = ax.imshow(disp, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("合成深度图示例（黑色 = 空洞）：上排为反光压板，中间为标签", loc="left", fontsize=11, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("深度（mm）", color=INK2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary, levels, table, yaws, yaw_res, path, args):
    lines = ["# 仿真结果", "",
             f"随机种子 {args.seed}，每个条件 {args.trials} 次，每次 {args.frames} 帧。合成场景见 `synthetic_d435.py`。", "",
             "## 总体（随机工况：距离 30–40 cm、偏航 ±8°、标签空洞 0–90%、压板空洞 70–98%）", "",
             "| 方法 | 单次成功率 | 误差中位数 (mm) | 95% 分位误差 (mm) | 误差 > 5 mm 且未报错的比例 |",
             "| --- | --- | --- | --- | --- |"]
    for m in METHODS:
        s = summary[m]
        med = "—" if s["median_err_mm"] is None else "%.2f" % s["median_err_mm"]
        p95 = "—" if s["p95_err_mm"] is None else "%.2f" % s["p95_err_mm"]
        lines.append(f"| {METHOD_LABELS[m]} | {s['success_rate'] * 100:.1f}% | {med} | {p95} | {s['silent_wrong_rate'] * 100:.1f}% |")
    pl = summary["plane"]
    lines += ["",
              f"面板平面方法中，{pl['share_plane'] * 100:.1f}% 由平面给出结果，{pl['share_roi_fallback'] * 100:.1f}% 回退到 ROI 中位数。"
              f"单次定位耗时中位数 {pl.get('median_time_ms', float('nan')):.0f} ms（仿真机器，960×540）。",
              f"深度方向误差落在 3σ 内的比例：{pl.get('coverage_3sigma_depth', float('nan')) * 100:.1f}%"
              f"（σ 中位数 {pl.get('median_sigma_normal_mm', float('nan')):.2f} mm）。", "",
              "## 成功率与标签空洞比例", "",
              "| 标签空洞 | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |",
              "| --- | " + " | ".join("---" for _ in METHODS) + " |"]
    for i, lv in enumerate(levels):
        lines.append(f"| {lv * 100:.0f}% | " + " | ".join("%.0f%%" % (table[m][i] * 100) for m in METHODS) + " |")
    lines += ["", "## 操作点误差与面板偏航（中位数，mm）", "",
              "| 偏航 | 原程序 | 新定位 + 固定方向 | 新定位 + 法向修正 |", "| --- | --- | --- | --- |"]
    for i, yw in enumerate(yaws):
        lines.append(f"| {yw}° | {yaw_res['old_pixel_nominal'][i][1]:.1f} | {yaw_res['plane_nominal'][i][1]:.1f} | "
                     f"{yaw_res['plane_corrected'][i][1]:.1f} |")
    lines += ["", "说明：这是简化的合成场景，用于验证代码和比较方法的相对优劣。论文中的数字必须来自真机实验。", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--plots-only", action="store_true", help="只用已有的 summary.json 重新作图")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.plots_only:
        with open(os.path.join(args.out, "summary.json"), encoding="utf-8") as f:
            data = json.load(f)
        plot_holes(np.array(data["holes"]["levels"]), data["holes"]["success"],
                   os.path.join(args.out, "fig_success_vs_holes.png"))
        plot_yaw(np.array(data["yaw"]["yaw_deg"]), data["yaw"]["op_err_mm_q25_med_q75"],
                 os.path.join(args.out, "fig_error_vs_yaw.png"))
        plot_example(np.random.default_rng(data["args"]["seed"] + 1), os.path.join(args.out, "fig_example_frame.png"))
        return
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    overall = [run_trial(random_params(rng), rng, args.frames) for _ in range(args.trials * 4)]
    summary = summarize(overall)
    levels, table, errs = experiment_holes(rng, args.trials, args.frames)
    yaws, yaw_res = experiment_yaw(rng, args.trials, args.frames)

    out = {"args": vars(args), "overall": summary,
           "holes": {"levels": levels.tolist(), "success": table, "median_err_mm": errs},
           "yaw": {"yaw_deg": yaws.tolist(), "op_err_mm_q25_med_q75": yaw_res},
           "elapsed_s": time.time() - t0}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    write_markdown(summary, levels, table, yaws, yaw_res, os.path.join(args.out, "summary.md"), args)
    if not args.no_plots:
        plot_holes(levels, table, os.path.join(args.out, "fig_success_vs_holes.png"))
        plot_yaw(yaws, yaw_res, os.path.join(args.out, "fig_error_vs_yaw.png"))
        plot_example(np.random.default_rng(args.seed + 1), os.path.join(args.out, "fig_example_frame.png"))
    print(open(os.path.join(args.out, "summary.md"), encoding="utf-8").read())
    print("耗时 %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
