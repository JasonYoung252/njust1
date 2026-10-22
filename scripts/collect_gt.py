#!/usr/bin/env python3
"""
真值采集：把视觉估计 p_hat 与 TCP 触碰得到的真值 p_true 配对记录，
得到"系统误差"样本，供 robust_opt.fit_from_data 使用。

前提：
  * target_position 节点已启动（roslaunch njust rm_63_ts.launch，或单独 rosrun）。
  * 末端装尖头工具，并已知尖头相对 /rm_driver/ArmCurrentState 所报位姿的偏移 tip_offset
    （在末端坐标系下，m）。若控制器里已把工具坐标系设到尖头，tip_offset 取 [0,0,0]。

用法：
  rosrun njust collect_gt.py _target:=U4C63V _direction:=0 _n_frames:=30 _tip_offset:="[0.0, 0.0, 0.0]"

每一轮：
  1. 把机械臂放到一个观测位姿（每轮换一换距离和角度），按回车 → 调 /pose 得到 p_hat；
  2. 示教把尖头对准【标签文字框中心】（OCR 定位的就是这个点），按回车 → 读 TCP 得到 p_true；
  3. 自动写入 <log_dir>/gt.csv，打印任务系误差。输入 q 回车结束。
建议 10~15 轮。测的是标签中心而不是压板：标签到压板的偏移是确定的几何量，不属于随机误差。
"""
import os
import csv
import time

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from njust.srv import pose
from rm_msgs.msg import ArmState, GetArmState_Command

GT_FIELDS = ["trial", "stamp", "target", "direction", "n_frames",
             "obs_ee_x", "obs_ee_y", "obs_ee_z",
             "hat_x", "hat_y", "hat_z", "true_x", "true_y", "true_z",
             "err_task_x", "err_task_y", "err_task_z", "dist"]


def base_to_task(v, direction):
    """基座系 -> 任务系（arm_control.dxy 的逆），z 不变。"""
    a = np.pi * (int(direction) % 10) / 2.0
    c, s = np.cos(a), np.sin(a)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])


class GTCollector:
    def __init__(self):
        rospy.init_node("collect_gt")
        self.target = rospy.get_param("~target")
        self.direction = int(rospy.get_param("~direction", 0))
        self.n_frames = int(rospy.get_param("~n_frames", 30))
        self.tip_offset = np.array(rospy.get_param("~tip_offset", [0.0, 0.0, 0.0]), dtype=float)
        self.log_dir = os.path.expanduser(rospy.get_param("~log_dir", "~/robust_opt_logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.path = os.path.join(self.log_dir, "gt.csv")

        # 让视觉节点按多帧测量，并在日志里记下方向；结束时恢复原值
        self._saved = {k: rospy.get_param(k, None) for k in ("/target_position/n_frames", "/robust_opt/direction")}
        rospy.set_param("/target_position/n_frames", self.n_frames)
        rospy.set_param("/robust_opt/direction", self.direction)

        self.ee = None
        rospy.Subscriber("/rm_driver/ArmCurrentState", ArmState, self._state_cb)
        self.state_pub = rospy.Publisher("/rm_driver/GetArmState_Cmd", GetArmState_Command, queue_size=10)
        self.client = rospy.ServiceProxy("/pose", pose)
        rospy.sleep(1.0)

    def _state_cb(self, msg):
        p, q = msg.Pose.position, msg.Pose.orientation
        self.ee = (np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))

    def read_ee(self):
        """主动请求一次状态并等待回调。返回 (位置, 四元数 xyzw)。"""
        self.ee = None
        cmd = GetArmState_Command()
        cmd.command = "get_current_arm_state"
        self.state_pub.publish(cmd)
        t0 = time.time()
        while self.ee is None and time.time() - t0 < 3.0 and not rospy.is_shutdown():
            rospy.sleep(0.05)
        if self.ee is None:
            raise RuntimeError("3 s 内没有收到 /rm_driver/ArmCurrentState")
        return self.ee

    def tip_position(self):
        pos, quat = self.read_ee()
        return pos + R.from_quat(quat).as_matrix() @ self.tip_offset

    def next_trial(self):
        if not os.path.exists(self.path):
            return 0
        with open(self.path) as f:
            return sum(1 for _ in f) - 1

    def run(self):
        print(f"目标标签 {self.target}，方向 {self.direction}，每次观测 {self.n_frames} 帧，记录到 {self.path}")
        trial = self.next_trial()
        while not rospy.is_shutdown():
            if input(f"\n[第 {trial} 轮] 机械臂到观测位姿后按回车（q 退出）: ").strip().lower() == "q":
                break
            obs_ee, _ = self.read_ee()
            rospy.wait_for_service("/pose")
            hat = np.array(self.client(self.target).pose[:3], dtype=float)
            if not np.any(hat):
                print("视觉没有找到标签，本轮作废")
                continue
            print(f"p_hat = {np.round(hat, 4)}")

            if input("示教把尖头对准标签文字框中心后按回车（s 跳过本轮）: ").strip().lower() == "s":
                continue
            true = self.tip_position()
            err = base_to_task(hat - true, self.direction)
            dist = float(np.linalg.norm(hat - obs_ee))
            print(f"p_true = {np.round(true, 4)}；任务系误差 (mm) = {np.round(err * 1000, 2)}；观测距离 ≈ {dist:.3f} m")

            new_file = not os.path.exists(self.path)
            with open(self.path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=GT_FIELDS)
                if new_file:
                    w.writeheader()
                w.writerow({"trial": trial, "stamp": f"{time.time():.3f}", "target": self.target,
                            "direction": self.direction, "n_frames": self.n_frames,
                            "obs_ee_x": obs_ee[0], "obs_ee_y": obs_ee[1], "obs_ee_z": obs_ee[2],
                            "hat_x": hat[0], "hat_y": hat[1], "hat_z": hat[2],
                            "true_x": true[0], "true_y": true[1], "true_z": true[2],
                            "err_task_x": err[0], "err_task_y": err[1], "err_task_z": err[2], "dist": dist})
            trial += 1
        print("结束")

    def restore(self):
        for k, v in self._saved.items():
            if v is None:
                if rospy.has_param(k):
                    rospy.delete_param(k)
            else:
                rospy.set_param(k, v)


if __name__ == "__main__":
    collector = GTCollector()
    try:
        collector.run()
    finally:
        collector.restore()
