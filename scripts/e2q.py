import numpy as np
from robotic_arm import *
import math

def euler_to_quaternion(rx, ry, rz):
    """
    将欧拉角（ZYX内旋顺序）转换为四元数

    参数:
    rx, ry, rz: 欧拉角（弧度），分别对应绕X、Y、Z轴的旋转角度
                按照ZYX内旋顺序：先绕Z轴旋转rz，再绕Y轴旋转ry，最后绕X轴旋转rx

    返回:
    quaternion: 四元数 [qx, qy, qz, qw]，其中qw为实部，qx, qy, qz为虚部
    """

    # 计算各轴旋转角度的一半
    half_rx = rx / 2.0
    half_ry = ry / 2.0
    half_rz = rz / 2.0

    # 计算三角函数值
    cos_rx = np.cos(half_rx)
    sin_rx = np.sin(half_rx)
    cos_ry = np.cos(half_ry)
    sin_ry = np.sin(half_ry)
    cos_rz = np.cos(half_rz)
    sin_rz = np.sin(half_rz)

    # 根据ZYX内旋顺序的四元数公式计算
    qw = cos_rx * cos_ry * cos_rz + sin_rx * sin_ry * sin_rz
    qx = sin_rx * cos_ry * cos_rz - cos_rx * sin_ry * sin_rz
    qy = cos_rx * sin_ry * cos_rz + sin_rx * cos_ry * sin_rz
    qz = cos_rx * cos_ry * sin_rz - sin_rx * sin_ry * cos_rz

    return np.array([qx, qy, qz, qw])


def pose_to_quaternion(x, y, z, rx, ry, rz):
    """
    将机械臂末端位姿（位置+欧拉角）转换为位置+四元数表示

    参数:
    x, y, z: 笛卡尔坐标位置
    rx, ry, rz: 欧拉角（弧度），ZYX内旋顺序

    返回:
    pose: [x, y, z, qx, qy, qz, qw] - 位置和四元数
    """
    position = np.array([x, y, z])
    quaternion = euler_to_quaternion(rx, ry, rz)

    return np.concatenate([position, quaternion])

def dxy(x, y, direction):
    dx = x * np.cos(np.pi*direction/2) + y * np.sin(np.pi*direction/2)
    dy = - x * np.sin(np.pi*direction/2) + y * np.cos(np.pi*direction/2)
    return dx, dy


# 示例使用
if __name__ == "__main__":
    # 示例：位置和欧拉角
    x, y, z = 1.0, 2.0, 3.0  # 位置 (m)
    rx, ry, rz = 0, -1.57, -2.355  # 欧拉角 (rad)

    print("输入:")
    print(f"位置: x={x}, y={y}, z={z}")
    print(f"欧拉角: rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f} (弧度)")
    print(f"欧拉角: rx={np.degrees(rx):.2f}°, ry={np.degrees(ry):.2f}°, rz={np.degrees(rz):.2f}°")

    # 转换为四元数
    quaternion = euler_to_quaternion(rx, ry, rz)
    full_pose = pose_to_quaternion(x, y, z, rx, ry, rz)

    print("\n输出:")
    print(f"四元数: qx={quaternion[0]:.4f}, qy={quaternion[1]:.4f}, qz={quaternion[2]:.4f}, qw={quaternion[3]:.4f}")
    # print(f"完整位姿: {full_pose}")

    # 验证四元数单位性质
    norm = np.linalg.norm(quaternion)
    print(f"\n四元数模长: {norm:.6f} (应该接近1.0)")


