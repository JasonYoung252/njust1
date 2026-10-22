"""坐标系工具：基座系 <-> 任务系。

arm_control.dxy(x, y) 把任务系平面位移 (x, y) 转到基座系：
    dx =  x*cos(a) + y*sin(a)
    dy = -x*sin(a) + y*cos(a),    a = pi*direction/2
这里给出它的逆变换，用于把基座系下的误差转到任务系。z 轴不变。
"""
import numpy as np


def _angle(direction):
    return np.pi * (int(direction) % 10) / 2.0


def task_to_base(vec, direction):
    """任务系向量 -> 基座系向量，与 arm_control.dxy 完全一致。vec 形状 (..., 3)。"""
    v = np.asarray(vec, dtype=float)
    a = _angle(direction)
    c, s = np.cos(a), np.sin(a)
    out = v.copy()
    out[..., 0] = v[..., 0] * c + v[..., 1] * s
    out[..., 1] = -v[..., 0] * s + v[..., 1] * c
    return out


def base_to_task(vec, direction):
    """基座系向量 -> 任务系向量（task_to_base 的逆）。vec 形状 (..., 3)。"""
    v = np.asarray(vec, dtype=float)
    a = _angle(direction)
    c, s = np.cos(a), np.sin(a)
    out = v.copy()
    out[..., 0] = v[..., 0] * c - v[..., 1] * s
    out[..., 1] = v[..., 0] * s + v[..., 1] * c
    return out
