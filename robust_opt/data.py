"""读取真机日志（target_position.py 的 vision_*.csv 与 collect_gt.py 的 gt.csv）。只依赖标准库与 numpy。"""
import csv
import glob
from collections import defaultdict

import numpy as np

from .frames import base_to_task


def _rows(paths):
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(p)) or [p])
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)


def load_jitter(vision_paths, min_frames=10, default_direction=None):
    """静态重复测量 -> 逐帧残差（任务系，m）。

    只使用帧数 >= min_frames 的测量组（session）；残差 = 逐帧基座坐标 - 本组中位数，再转到任务系。
    日志里 direction = -1（未知）的组：若给了 default_direction 就用它，否则跳过。
    返回 (残差数组 (J,3), 使用的组数)。
    """
    groups = defaultdict(list)
    for r in _rows(vision_paths):
        groups[(r["session"], r["target"])].append(r)
    res, used = [], 0
    for rows in groups.values():
        if len(rows) < min_frames:
            continue
        d = int(rows[0]["direction"])
        if d < 0:
            if default_direction is None:
                continue
            d = default_direction
        p = np.array([[float(r["base_x"]), float(r["base_y"]), float(r["base_z"])] for r in rows])
        res.append(base_to_task(p - np.median(p, axis=0), d))
        used += 1
    if not res:
        return np.zeros((0, 3)), 0
    return np.vstack(res), used


def load_systematic(gt_paths):
    """真值实验 -> 每次观测的误差 m = p_hat - p_true（任务系，m）与观测距离。"""
    err, dist = [], []
    for r in _rows(gt_paths):
        err.append([float(r["err_task_x"]), float(r["err_task_y"]), float(r["err_task_z"])])
        dist.append(float(r.get("dist") or "nan"))
    return np.array(err).reshape(-1, 3), np.array(dist)
