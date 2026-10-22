import numpy as np
from scipy.spatial.transform import Rotation as R

# 相机坐标系到机械臂末端坐标系的手眼标定结果（眼在手上）
# 旋转矩阵
HAND_EYE_ROTATION = np.array([[-0.0145146, -0.99980617, 0.01330249],
                              [ 0.9998913, -0.01454777, -0.00239994],
                              [ 0.0025930,  0.01326621,  0.99990864]])
# 平移向量（m）
HAND_EYE_TRANSLATION = np.array([0.07990557, -0.04388621, -0.19574536])
# HAND_EYE_TRANSLATION = np.array([0.07990557, -0.04388621, -0.18474536])
# 注：上面两组 z 值相差 11 mm，说明手眼标定存在系统误差，建议重新标定并记录标定残差。


def camera_to_end_effector_matrix():
    T = np.eye(4)
    T[:3, :3] = HAND_EYE_ROTATION
    T[:3, 3] = HAND_EYE_TRANSLATION
    return T


def end_effector_to_base_matrix(x1, y1, z1, rx, ry, rz):
    """机械臂末端位姿（位置 m，欧拉角 rad，xyz 外旋）-> 齐次变换矩阵。"""
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', [rx, ry, rz], degrees=False).as_matrix()
    T[:3, 3] = [x1, y1, z1]
    return T


def camera_to_base_matrix(x1, y1, z1, rx, ry, rz):
    """相机坐标系 -> 机械臂基座坐标系的齐次变换矩阵。"""
    return end_effector_to_base_matrix(x1, y1, z1, rx, ry, rz).dot(camera_to_end_effector_matrix())


def convert(x, y, z, x1, y1, z1, rx, ry, rz):
    """
    使用深度相机识别到的物体坐标（x, y, z）和机械臂末端的位姿（x1,y1,z1,rx,ry,rz）
    来计算物体在机械臂基坐标下的位置（x, y, z）

    Args:
        x : 相机坐标系下物体位置x
        y : 相机坐标系下物体位置y
        z : 相机坐标系下物体位置z
        x1 : 机械臂末端位姿 x
        y1 : 机械臂末端位姿 y
        z1 : 机械臂末端位姿 z
        rx : 机械臂末端位姿 rx
        ry : 机械臂末端位姿 ry
        rz : 机械臂末端位姿 rz

    Returns:
        _type_: 物体在机械臂基坐标系下的位置 x y z
    """
    T = camera_to_base_matrix(x1, y1, z1, rx, ry, rz)
    obj_base = T.dot(np.array([x, y, z, 1.0]))
    x, y, z = obj_base[:3]
    return x, y, z


def rotate_to_base(vec, x1, y1, z1, rx, ry, rz):
    """把相机坐标系下的方向向量（如面板法向）转到基座坐标系，只做旋转。"""
    T = camera_to_base_matrix(x1, y1, z1, rx, ry, rz)
    return T[:3, :3].dot(np.asarray(vec, dtype=float))


def covariance_to_base(cov_cam, x1, y1, z1, rx, ry, rz):
    """相机坐标系下的 3×3 协方差 -> 基座坐标系。"""
    Rm = camera_to_base_matrix(x1, y1, z1, rx, ry, rz)[:3, :3]
    return Rm.dot(np.asarray(cov_cam, dtype=float)).dot(Rm.T)
