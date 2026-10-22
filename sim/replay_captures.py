#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线复现：读取 target_position 在 save_dir 中保存的采集数据，用不同参数重新定位，并统计重复性。

    python3 sim/replay_captures.py /path/to/save_dir
    python3 sim/replay_captures.py /path/to/save_dir --plane-radius 0.05 --csv replay.csv

每个标签输出：样本数、成功率、基座坐标的标准差（mm，反映重复性）、σ 估计的中位数。
真机上对同一标签、同一机械臂位姿重复采集 20–30 次，就能得到论文里“定位重复性”的数据。
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from depth_localizer import CameraIntrinsics, DepthLocalizer, LocalizerConfig, box_from_polygon  # noqa: E402
import convert  # noqa: E402


def load_captures(folder):
    for meta_path in sorted(glob.glob(os.path.join(folder, "*_meta.json"))):
        stem = meta_path[:-len("_meta.json")]
        depth_path = stem + "_depth.npz"
        if not os.path.exists(depth_path):
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        depth = np.load(depth_path)["depth_mm"].astype(np.float32) * 0.001
        yield stem, meta, depth


def replay(folder, cfg):
    rows = []
    for stem, meta, depth in load_captures(folder):
        fx, fy, cx, cy = meta["intrinsics"]
        loc = DepthLocalizer(CameraIntrinsics(fx, fy, cx, cy), cfg)
        box = box_from_polygon(meta["polygon"]) if meta.get("polygon") else None
        res = loc.localize(depth, tuple(meta["center"]), box)
        row = {"capture": os.path.basename(stem), "target": meta["target"], "status": res.status,
               "method": res.method, "message": res.message,
               "sigma_normal_mm": res.sigma_normal_m * 1000, "roi_valid_ratio": res.roi_valid_ratio}
        if res.ok:
            row.update(dict(zip(("x_cam", "y_cam", "z_cam"), [float(a) for a in res.point_cam])))
            if meta.get("arm_pose"):
                base = convert.convert(*res.point_cam, *meta["arm_pose"])
                row.update(dict(zip(("x_base", "y_base", "z_base"), [float(a) for a in base])))
        rows.append(row)
    return rows


def summarize(rows):
    out = {}
    for target in sorted(set(r["target"] for r in rows)):
        rs = [r for r in rows if r["target"] == target]
        ok = [r for r in rs if r["status"] > 0]
        key = "x_base" if ok and "x_base" in ok[0] else "x_cam"
        cols = ("x_base", "y_base", "z_base") if key == "x_base" else ("x_cam", "y_cam", "z_cam")
        pts = np.array([[r[c] for c in cols] for r in ok]) if ok else np.zeros((0, 3))
        out[target] = {
            "n": len(rs), "success_rate": len(ok) / float(len(rs)),
            "frame": "base" if key == "x_base" else "camera",
            "std_mm": (pts.std(axis=0, ddof=1) * 1000).tolist() if len(pts) > 1 else None,
            "median_sigma_normal_mm": float(np.median([r["sigma_normal_mm"] for r in ok])) if ok else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--plane-radius", type=float, default=None)
    ap.add_argument("--inlier-thresh", type=float, default=None)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    cfg = LocalizerConfig()
    if args.plane_radius is not None:
        cfg.plane_radius_m = args.plane_radius
    if args.inlier_thresh is not None:
        cfg.plane_inlier_thresh_m = args.inlier_thresh
    rows = replay(args.folder, cfg)
    if not rows:
        print("没有找到采集数据（*_meta.json + *_depth.npz）")
        return
    if args.csv:
        keys = sorted(set(k for r in rows for k in r))
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    for target, s in summarize(rows).items():
        std = "—" if s["std_mm"] is None else "/".join("%.2f" % v for v in s["std_mm"])
        print(f"{target}: 样本 {s['n']}，成功率 {s['success_rate']:.0%}，{s['frame']} 坐标标准差 xyz = {std} mm，"
              f"σn 中位数 {s['median_sigma_normal_mm']}")


if __name__ == "__main__":
    main()
