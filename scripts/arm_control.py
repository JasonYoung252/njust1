#!/usr/bin/env python3
import math
import copy
from e2q import euler_to_quaternion
import pose_utils
from exception import TargetNotFoundException
import rospy
import threading
import numpy as np
from robotic_arm import *
from njust.msg import result
from njust.srv import task, taskResponse, pose, poseResponse
from rm_msgs.msg import ArmState, Gripper_Pick, Gripper_Set, GetArmState_Command, MoveJ, MoveJ_P, MoveL, Plan_State

class arm_control:
    def __init__(self):
        rospy.init_node('arm_control')
        rospy.sleep(1.0)
        
        # 存储参数
        self.current_joint = [0, 0, 0, 0, 0, 0]
        self.current_pose = [0, 0, 0, 0, 0, 0, 0]
        self.target_pose = [0, 0, 0, 0, 0, 0, 0]
        self.current_request = None
        self.action = False
        self.arm_dof = 6
        self.arrival_flag = 0
        self.get = True

        # 定位重试与偏航修正参数（rosparam，节点私有命名空间）
        self.max_pose_attempts = int(rospy.get_param('~max_pose_attempts', 10))
        self.use_yaw_correction = bool(rospy.get_param('~use_yaw_correction', False))
        self.max_yaw_correction_deg = float(rospy.get_param('~max_yaw_correction_deg', 10.0))
        self.yaw_delta = 0.0          # 实测面板偏航相对名义方向的修正量（rad），每个任务开始时清零
        self.target_info = None       # 最近一次定位的解析结果（含不确定性）

        # ip_address = "192.168.123.88"
        # model = RML63_II
        # self.robot = Arm(model, ip_address)  # 直接传递字符串，不需要转换
        # if self.robot.API_Version():
        #     print(f"成功连接到机械臂IP地址{ip_address}")
        # else:
        #     print("连接机械臂失败。")

        # 订阅机械臂状态
        self.Arm_Base_State_sub = rospy.Subscriber("/rm_driver/ArmCurrentState", ArmState, self.Get_Arm_State_Callback)
        # 发布夹爪控制命令
        self.gripper_pub = rospy.Publisher('/rm_driver/Gripper_Pick', Gripper_Pick, queue_size=10)
        # 发布夹爪到设定位置命令
        self.gripper_pose_pub = rospy.Publisher('/rm_driver/Gripper_Set', Gripper_Set, queue_size=10)

        self.moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=10)
        self.moveJ_P_pub = rospy.Publisher("/rm_driver/MoveJ_P_Cmd", MoveJ_P, queue_size=10)
        self.moveL_pub = rospy.Publisher("/rm_driver/MoveL_Cmd", MoveL, queue_size=10)

        self.result_pub = rospy.Publisher("/result_info", result, queue_size=10)

        rospy.sleep(2.0)

        self.planState_sub = rospy.Subscriber("/rm_driver/Plan_State", Plan_State, self.planStateCallback)

        self.command = GetArmState_Command()
        self.command.command = "get_current_arm_state"
        self.test_Get_Arm_Base_State_pub = rospy.Publisher("/rm_driver/GetArmState_Cmd", GetArmState_Command, queue_size=10)

        self.service = rospy.Service('/task', task, self.taskCallback)
        self.client = rospy.ServiceProxy('/pose', pose)

        # 多线程
        self.action_thread = threading.Thread(target=self.action_thread_func, daemon=True)
        self.action_thread.start()

    def run(self):
        rospy.sleep(0.5)
        
        self.set_gripper(0)
        self.init_position(0)
        print("机械臂初始化完成")

    def action_thread_func(self):
        """
        多线程 分配任务
        button(按钮) switch(压板) door1(屏柜门正面) door2(屏柜门背面)
        """
        while not rospy.is_shutdown():
            if self.action:
                try:
                    self._dispatch_task()
                except TargetNotFoundException as e:
                    self.handle_target_not_found(e)
                except Exception as e:
                    rospy.logerr(f"任务执行出错：{e}")
                    self.result_pub.publish(f"fail: {e}")
                finally:
                    self.action = False
            rospy.sleep(0.1)  # 防止CPU占用过高

    def _dispatch_task(self):
        """按任务类型执行（原 action_thread_func 中的分支，逻辑未改）"""
        if self.current_request.type == 'd':
            # 转方向
            self.init_position(self.current_request.direction)
            self.result_pub.publish("success")

        if self.current_request.type == 'p':
            # 
            self.get_target_pose()
            self.result_pub.publish("success")
        if self.current_request.type == 'button':
            # 按钮
            self.init_position(self.current_request.direction)
            self.position_dxyz(0.35, 0, -0.1)
            # self.position_dxyz(0.1, 0, 0)
            self.get_target_pose()
            self.control_button()
            self.init_position(self.current_request.direction)
            self.result_pub.publish("success")

        if self.current_request.type == 'knob':
            # 旋钮 未完成)
            self.init_position(self.current_request.direction)
            self.position_dxyz(0.3, -0.12, -0.25)
            self.get_target_pose()
            self.control_knob()
            self.init_position(self.current_request.direction)

        if self.current_request.type == 'switch':
            i = 0
            self.init_position(self.current_request.direction)
            if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                # 压板投入
                self.position_dxyz(0.35, 0, -0.1)

                self.get_target_pose()
                self.control_switch_open(0.033)
                self.init_position(self.current_request.direction)
                self.result_pub.publish("success")
            if self.current_request.now_state == 1 and self.current_request.next_state == 0:
                # 压板退出
                self.position_dxyz(0.35, 0, -0.1)
                self.get_target_pose()
                self.control_switch_close()
                self.init_position(self.current_request.direction)
                self.result_pub.publish("success")

        if self.current_request.type == 'door':
            if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                # 右拉门 开
                self.init_position(10)
                self.position_dxyz(0.35, -0.05, -0.3)
                # self.position_dxyz(0.14, 0, 0)
                self.get_target_pose()
                self.control_door_right_pull()

        if self.current_request.type == 'door1':
            if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                # 柜门正面 开
                self.init_position(self.current_request.direction+10)
                if self.current_request.direction == 1:
                    self.position_dxyz(0.25, 0.15, -0.25)
                    self.get_target_pose()
                    self.control_door1_open_right()
                self.init_position(self.current_request.direction+10)
                self.result_pub.publish("success")
            if self.current_request.now_state == 1 and self.current_request.next_state == 0:
                # 柜门正面 关
                self.init_position(self.current_request.direction+10)
                if self.current_request.direction == 1:
                    self.position_dxyz(0.25, 0.15, -0.25)
                    self.get_target_pose()
                    self.init_position(self.current_request.direction+10)
                    self.init_position(12)
                    self.control_door1_close_right()
                    self.init_position(self.current_request.direction+10)
                    self.position_dxyz(0.25, 0.15, -0.25)
                    self.get_target_pose()
                    self.control_door1_close2_right()
                    self.init_position(self.current_request.direction+10)
                    self.result_pub.publish("success")
                if self.current_request.direction == 3:
                    self.control_door1_close1_left()
                # self.position_dxyz(0.05, 0, -0.12)
                # if not self.get_target_pose():
                #     continue
                # self.control_door1_close2()

        if self.current_request.type == 'door2':
            if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                self.position_dxyz(0.2, 0, -0.2)
                self.get_target_pose()
                self.control_door2_open2_0()

        if self.current_request.type == 'door3':
            # 淮安 防火门
            if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                self.init_position(11)
                self.position_dxyz(0.3, 0, -0.3)
                self.get_target_pose()
                self.control_door3_open()
                self.init_position(11)
            if self.current_request.now_state == 1 and self.current_request.next_state == 0:
                self.init_position(12)
                self.control_door3_close()
                self.get_target_pose()
                self.control_door3_close2()
                self.init_position(12)
                self.control_door3_close3()

    def handle_target_not_found(self, e):
        """多次定位失败：沿原方向后退 10 cm，上报失败，不再朝无效目标运动。"""
        rospy.logerr(f"目标定位失败，任务中止：{e}")
        try:
            self.yaw_delta = 0.0
            self.position_dxyz(-0.1, 0, 0)
        except Exception as move_err:
            rospy.logerr(f"后退失败：{move_err}")
        self.result_pub.publish(f"fail: target {self.current_request.target} not found")
    
    def taskCallback(self, req):
        """
        string target   标签
        string type   类型   button(按钮) switch(压板) door2(屏柜门)
        uint8 direction   方向 前右后左 - 0123
        uint8 region   区域
        float32[] offset   开关和标签的偏移  面朝标签 前为x轴,左为y轴,上为z轴
        uint8 now_state   开关当前状态
        uint8 next_state   开关目标状态
        """
        self.current_request = req
        self.yaw_delta = 0.0
        self.target_info = None
        print(f"收到任务:target={req.target}, type={req.type}, direction={req.direction}, region={req.region}, offset={req.offset}, now_state={req.now_state}, next_state={req.next_state}")
        self.action = True
        return taskResponse("OK")

    def planStateCallback(self, msg):
        if msg.state:
            self.arrival_flag +=  1
        else:
            rospy.logerr("Plan State Fail arrival_flag is %d", self.arrival_flag)

    def Get_Arm_State_Callback(self, msg):
        """机械臂状态回调函数"""
        self.current_joint = list(msg.joint)
        self.current_pose[0] = msg.Pose.position.x
        self.current_pose[1] = msg.Pose.position.y
        self.current_pose[2] = msg.Pose.position.z
        self.current_pose[3] = msg.Pose.orientation.x
        self.current_pose[4] = msg.Pose.orientation.y
        self.current_pose[5] = msg.Pose.orientation.z
        self.current_pose[6] = msg.Pose.orientation.w
        self.get = False

    def get_target_pose(self, x = 0, y = 0, z = 0):
        """
        通过相机获取目标状态。
        原程序在定位失败时无限重试；现在最多重试 max_pose_attempts 次，仍失败则抛出 TargetNotFoundException，
        由 action_thread_func 中止任务，避免机械臂朝无效坐标运动。
        """
        self.get_state()
        for attempt in range(1, self.max_pose_attempts + 1):
            rospy.wait_for_service('/pose')
            self.target_pose = self.client(self.current_request.target)
            info = pose_utils.unpack_pose(self.target_pose.pose)
            far_enough = info["x"] ** 2 + info["y"] ** 2 > self.current_pose[0] ** 2 + self.current_pose[1] ** 2
            if info["valid"] and far_enough:
                self.target_info = info
                self.update_yaw_correction(info)
                if info["extended"]:
                    rospy.loginfo(f"定位有效（第{attempt}次）：status={info['status']}, "
                                  f"σn={info['sigma_normal'] * 1000:.2f} mm, σlat={info['sigma_lateral'] * 1000:.2f} mm")
                return True
            rospy.logwarn(f"第{attempt}/{self.max_pose_attempts}次定位无效：{list(self.target_pose.pose)[:3]}")
            if x == 0 and y == 0 and z == 0:
                rospy.sleep(0.2)
            else:
                self.position_dxyz(x, y, z)
        raise TargetNotFoundException(f"标签 {self.current_request.target} 连续 {self.max_pose_attempts} 次定位无效")

    def update_yaw_correction(self, info):
        """根据实测面板朝向计算偏移量的偏航修正（默认关闭，只记录日志）。"""
        delta, ok = pose_utils.yaw_correction(self.current_request.direction, info.get("facing_yaw"),
                                              self.max_yaw_correction_deg)
        if ok:
            rospy.loginfo(f"面板实测偏航 {np.degrees(delta):.2f}°（{'已启用修正' if self.use_yaw_correction else '未启用修正'}）")
        self.yaw_delta = delta if (ok and self.use_yaw_correction) else 0.0

    def get_state(self):
        """获取机械臂状态"""
        self.test_Get_Arm_Base_State_pub.publish(self.command)
        rospy.sleep(1.0)
        while self.get:
            pass
        self.get = True

    def close_gripper(self, force=200, speed=200):
        """控制夹爪闭合"""
        if not rospy.is_shutdown():
            gripper_cmd = Gripper_Pick()
            gripper_cmd.speed = speed  # 速度
            gripper_cmd.force = force  # 力度
            self.gripper_pub.publish(gripper_cmd)

    def set_gripper(self, position=230):
        """控制夹爪位置"""
        if not rospy.is_shutdown():
            gripper_cmd = Gripper_Set()
            gripper_cmd.position = position  # 位置
            self.gripper_pose_pub.publish(gripper_cmd)
    
    def dxy(self, x, y):
        """面朝标签坐标系（前 x、左 y）的偏移 -> 基座坐标系。yaw_delta=0 时与原程序完全一致。"""
        return pose_utils.rotate_offset(x, y, self.current_request.direction, self.yaw_delta)

    # -------------------------------------------功能函数-------------------------------------------
    def init_position(self, num):
        """初始位置 和 相对的区域"""
        if num == 0:
            """
            夹爪朝前
            x, y, z = 0.37, 0, 0.5
            rx, ry, rz = 3.14, -1.57, 0
            """
            if self.arrival_flag == 0:
                moveJ_BeginPose = MoveJ()
                moveJ_BeginPose.joint = [-0.002827, -0.801839, 2.048196, 0.010123, 0.324143, 3.122795]
                moveJ_BeginPose.speed = 0.15
                self.moveJ_pub.publish(moveJ_BeginPose)
            while self.arrival_flag == 0 and not rospy.is_shutdown():
                pass
            rospy.sleep(1.0)
            self.arrival_flag = 0
            print("到达初始姿态")
        
        if num == 1:
            """
            夹爪朝右
            x, y, z = 0, -0.37, 0.5
            rx, ry, rz = 1.57, -1.57, 0
            """
            if self.arrival_flag == 0:
                moveJ_BeginPose = MoveJ()
                moveJ_BeginPose.joint = [-1.573623, -0.801839, 2.048196, 0.010123, 0.324143, 3.122795]
                moveJ_BeginPose.speed = 0.15
                self.moveJ_pub.publish(moveJ_BeginPose)
            while self.arrival_flag == 0 and not rospy.is_shutdown():
                pass
            rospy.sleep(1.0)
            self.arrival_flag = 0
            print("到达右侧初始姿态")

        if num == 2:
            """
            夹爪朝后
            x, y, z = -0.5, 0.05, 0.48
            rx, ry, rz = 0, -1.57, 0
            """
            if self.arrival_flag == 0:
                moveJ_BeginPose = MoveJ()
                moveJ_BeginPose.joint = [-0.275587, 0.931552, -2.471892, 1.463511, -0.277263, -1.460073]
                moveJ_BeginPose.speed = 0.15
                self.moveJ_pub.publish(moveJ_BeginPose)
            while self.arrival_flag == 0 and not rospy.is_shutdown():
                pass
            rospy.sleep(1.0)
            self.arrival_flag = 0
            print("到达后方初始姿态")

        if num == 3:
            """
            夹爪朝左
            x, y, z = 0, 0.37, 0.5
            rx, ry, rz = 0, -1.57, -1.57
            """
            if self.arrival_flag == 0:
                moveJ_BeginPose = MoveJ()
                moveJ_BeginPose.joint = [1.567970, -0.801839, 2.048196, 0.010123, 0.324143, 3.122795]
                moveJ_BeginPose.speed = 0.15
                self.moveJ_pub.publish(moveJ_BeginPose)
            while self.arrival_flag == 0 and not rospy.is_shutdown():
                pass
            rospy.sleep(1.0)
            self.arrival_flag = 0
            print("到达左侧初始姿态")

        if num == 10:
            """
            夹爪朝前
            x, y, z = 0.37, 0, 0.5
            rx, ry, rz = 3.14, -1.57, 0
            """
            if self.arrival_flag == 0:
                moveJ_BeginPose = MoveJ()
                moveJ_BeginPose.joint = [-0.002827, -0.801839, 2.048196, 0.010123, 0.324143, 3.122795]
                moveJ_BeginPose.speed = 0.15
                self.moveJ_pub.publish(moveJ_BeginPose)
            while self.arrival_flag == 0 and not rospy.is_shutdown():
                pass
            rospy.sleep(1.0)
            self.arrival_flag = 0
            print("到达初始姿态")

        if num == 11:
            if self.arrival_flag == 0:
                moveJ_TargetPose =MoveJ()
                joint = [90, 67.046, -132.746, 0, -24.256, 0]
                joint_rad = [math.radians(angle) for angle in joint]
                moveJ_TargetPose.joint = joint_rad
                moveJ_TargetPose.speed = 0.1
                self.moveJ_pub.publish(moveJ_TargetPose)
            while self.arrival_flag == 0 and not rospy.is_shutdown():
                pass
            rospy.sleep(1.0)
            self.arrival_flag = 0

        if num == 12:
            if self.arrival_flag == 0:
                moveJ_TargetPose =MoveJ()
                joint = [0, 67.046, -132.746, 0, -24.256, 0]
                joint_rad = [math.radians(angle) for angle in joint]
                moveJ_TargetPose.joint = joint_rad
                moveJ_TargetPose.speed = 0.1
                self.moveJ_pub.publish(moveJ_TargetPose)
            while self.arrival_flag == 0 and not rospy.is_shutdown():
                pass
            rospy.sleep(1.0)
            self.arrival_flag = 0

        if num == 13:
            if self.arrival_flag == 0:
                moveJ_TargetPose =MoveJ()
                joint = [-90, 67.046, -132.746, 0, -24.256, 0]
                joint_rad = [math.radians(angle) for angle in joint]
                moveJ_TargetPose.joint = joint_rad
                moveJ_TargetPose.speed = 0.1
                self.moveJ_pub.publish(moveJ_TargetPose)
            while self.arrival_flag == 0 and not rospy.is_shutdown():
                pass
            rospy.sleep(1.0)
            self.arrival_flag = 0

    def position_dxyz(self, x, y, z):
        """姿态不变，位移相对距离"""
        if self.arrival_flag == 0:
            self.get_state()
            moveJ_P_TargetPose = MoveJ_P()
            dx, dy = self.dxy(x, y)
            moveJ_P_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.current_pose[2] + z
            moveJ_P_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        self.arrival_flag = 0

    # -------------------------------------------按钮-------------------------------------------
    def control_button(self):
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0

        if self.arrival_flag == i:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1])
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2]
            q = euler_to_quaternion(3.14, -1.57, 0)
            moveJ_P_TargetPose.Pose.orientation.x = q[0]
            moveJ_P_TargetPose.Pose.orientation.y = q[1]
            moveJ_P_TargetPose.Pose.orientation.z = q[2]
            moveJ_P_TargetPose.Pose.orientation.w = q[3]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.3
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0.072, 0)
            moveJ_P_TargetPose.Pose.position.x += dx
            moveJ_P_TargetPose.Pose.position.y += dy
            moveJ_P_TargetPose.speed = 0.2
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.071, 0)
            moveJ_P_TargetPose.Pose.position.x += dx
            moveJ_P_TargetPose.Pose.position.y += dy
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------旋钮-------------------------------------------
    def control_knob(self):
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        if self.arrival_flag == 0:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1])
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2]
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        
        if self.arrival_flag == 1:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += -np.pi/2
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper(250)
        rospy.sleep(2.0)

        if self.arrival_flag == 2:
            self.get_state()
            dx, dy = self.dxy(0.07, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 3:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += np.pi/2
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper(400)
        rospy.sleep(2.0)

        if self.arrival_flag == 4:
            self.get_state()
            dx, dy = self.dxy(-0.15, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 4 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper(0)
        rospy.sleep(2.0)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------压板投入-------------------------------------------
    def control_switch_open(self, l=0.036):
        # 压板 投入
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0
        # 下 转紧半圈
        if self.arrival_flag == i:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1]-0.002)
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2] - l
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # if self.arrival_flag == i:
        #     self.get_state()
        #     moveJ_TargetPose.joint = self.current_joint
        #     moveJ_TargetPose.joint[5] += -np.pi/3
        #     moveJ_TargetPose.speed = 0.1
        #     self.moveJ_pub.publish(moveJ_TargetPose)
        #     i += 1
        # while self.arrival_flag == i - 1 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        self.set_gripper(200)
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            dx, dy = self.dxy(0.065, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.close_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += np.pi/4
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper(100)
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            dx, dy = self.dxy(-0.02, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.close_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += -np.pi/4
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # 压板投入
        if self.arrival_flag == i:
            self.get_state()
            moveL_TargetPose.Pose.position.x = self.current_pose[0]
            moveL_TargetPose.Pose.position.y = self.current_pose[1]
            moveL_TargetPose.Pose.position.z = self.current_pose[2] -0.022
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0.038, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, 0.023)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            moveL_TargetPose.Pose.position.z += 0.038
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, -0.016)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, 0.02)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.07, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1]-0.0022)
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2] + 0.0018
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += -np.pi/2
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            dx, dy = self.dxy(0.065, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.close_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += np.pi
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            dx, dy = self.dxy(-0.1, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.close_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += -np.pi/2
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    def control_switch_open2_0(self, l=0.036):
        # 压板 投入
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0
        # 下 转紧半圈
        if self.arrival_flag == i:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1])
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2] - l
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            self.get_state()
            dx, dy = self.dxy(0.045, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # 压板投入
        if self.arrival_flag == i:
            self.get_state()
            moveL_TargetPose.Pose.position.x = self.current_pose[0]
            moveL_TargetPose.Pose.position.y = self.current_pose[1]
            moveL_TargetPose.Pose.position.z = self.current_pose[2] -0.022
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0.038, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, 0.023)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            moveL_TargetPose.Pose.position.z += 0.038
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, -0.016)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, 0.02)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.07, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1])
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2] + 0.0025
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += -np.pi/2
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            dx, dy = self.dxy(0.065, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.close_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += np.pi
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            dx, dy = self.dxy(-0.1, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.close_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += -np.pi/2
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------压板退出-------------------------------------------
    def control_switch_close(self):
        # 压板 退出
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        if self.arrival_flag == 0:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1])
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2] + 0.0025
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        
        if self.arrival_flag == 1:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += np.pi/2
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == 2:
            self.get_state()
            dx, dy = self.dxy(0.065, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.close_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == 3:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += -np.pi
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == 4:
            self.get_state()
            dx, dy = self.dxy(-0.04, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 4 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.close_gripper()
        rospy.sleep(2.0)

        if self.arrival_flag == 5:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += np.pi/2
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 5 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 6:
            self.get_state()
            dx, dy = self.dxy(0, -0.023)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2] - 0.015
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 6 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 7:
            dx, dy = self.dxy(0.045, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 7 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 8:
            dx, dy = self.dxy(0, 0.04)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 8 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 9:
            dx, dy = self.dxy(-0.1, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 9 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------门右拉-------------------------------------------
    def control_door_right_pull(self, l=0.56):
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        if self.arrival_flag == 0:
            # dx, dy = self.dxy(self.current_request.offset[0], self.current_request.offset[1])
            # self.get_state()
            # moveJ_P_TargetPose.Pose.position.x = self.current_pose[0] + dx
            # moveJ_P_TargetPose.Pose.position.y = self.current_pose[1] + dy
            # moveJ_P_TargetPose.Pose.position.z = self.current_pose[2] + self.current_request.offset[2]
            # moveJ_P_TargetPose.Pose.orientation.x = self.current_pose[3]
            # moveJ_P_TargetPose.Pose.orientation.y = self.current_pose[4]
            # moveJ_P_TargetPose.Pose.orientation.z = self.current_pose[5]
            # moveJ_P_TargetPose.Pose.orientation.w = self.current_pose[6]
            # moveJ_P_TargetPose.trajectory_connect = 0

            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1])
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2]
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        
        self.set_gripper(500)
        rospy.sleep(2.0)

        if self.arrival_flag == 1:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.08, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper(100)
        rospy.sleep(2.0)

        if self.arrival_flag == 2:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            theta = np.pi/6
            dx, dy = self.dxy(0, 0.07*(1-np.cos(theta)))
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += -0.07*np.sin(theta)
            next_pose[3] = -1.57
            next_pose[4] = -1.57 + theta
            next_pose[5] = -1.57
            self.robot.Movej_P_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        if self.arrival_flag == 3:
            next_pose = copy.deepcopy(current_pose)
            theta = np.pi/3
            dx, dy = self.dxy(0, 0.07*(1-np.cos(theta)))
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += -0.07*np.sin(theta)
            next_pose[3] = -1.57
            next_pose[4] = -1.57 + theta
            next_pose[5] = -1.57
            self.robot.Movej_P_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        if self.arrival_flag == 4:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            s=np.sqrt(0.055**2+l**2)
            alpha = np.arctan(0.055/l)
            theta = np.pi/8
            dx, dy = self.dxy(-s*np.sin(alpha+theta)+0.055, -s*(np.cos(alpha)-np.cos(alpha+theta)))
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[5] += theta
            self.robot.Movej_P_Cmd(next_pose, 5, 0, trajectory_connect=0, block=True)
        while self.arrival_flag == 4 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        if self.arrival_flag == 5:
            next_pose = copy.deepcopy(current_pose)
            theta = np.pi/4
            dx, dy = self.dxy(-s*np.sin(alpha+theta)+0.055, -s*(np.cos(alpha)-np.cos(alpha+theta)))
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[5] += theta/2
            self.robot.Movej_P_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
        while self.arrival_flag == 5 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        self.set_gripper(1000)
        rospy.sleep(2.0)

        if self.arrival_flag == 6:
            dx, dy = self.dxy(0, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += 0.07*np.sin(np.pi/5)
            self.robot.Movej_P_Cmd(next_pose, 5, 0, trajectory_connect=0, block=True)
        while self.arrival_flag == 6 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        if self.arrival_flag == 7:
            dx, dy = self.dxy(-0.15, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movej_P_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
        while self.arrival_flag == 7 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        self.set_gripper(0)
        rospy.sleep(2.0)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------屏柜门正面开-------------------------------------------
    def control_door1_open_left(self):
        # 开屏柜门 左拉 正面
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        if self.arrival_flag == 0:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1]) 
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2]
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            move0 = moveJ_P_TargetPose
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        
        # 按开关
        if self.arrival_flag == 1:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.078, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 2:
            dx, dy = self.dxy(-0.05, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # 转把手
        if self.arrival_flag == 3:
            dx, dy = self.dxy(0, -0.03)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 4:
            moveL_TargetPose.Pose.position.z +=  0.05
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 4 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 5:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.02, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 5 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 6:
            dx, dy = self.dxy(0, 0.07)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 6 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 7:
            moveL_TargetPose.Pose.position.z +=  0.025
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 7 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 8:
            moveL_TargetPose.Pose.position.z +=  -0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 8 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 9:
            dx, dy = self.dxy(-0.1, -0.2)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 9  and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    def control_door1_open_right(self):
        # 开屏柜门 右拉 正面
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        if self.arrival_flag == 0:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1]) 
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2]
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            move0 = moveJ_P_TargetPose
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        
        # 按开关
        if self.arrival_flag == 1:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.074, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 2:
            dx, dy = self.dxy(-0.05, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # 转把手
        if self.arrival_flag == 3:
            dx, dy = self.dxy(0, 0.03)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 4:
            moveL_TargetPose.Pose.position.z +=  0.05
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 4 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 5:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.02, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 5 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 6:
            dx, dy = self.dxy(0, -0.07)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 6 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 7:
            moveL_TargetPose.Pose.position.z +=  0.04
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 7 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 8:
            dx, dy = self.dxy(-0.1, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 8 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 9:
            moveL_TargetPose.Pose.position.z +=  -0.14
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 9 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 10:
            dx, dy = self.dxy(-0.1, 0.2)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 10  and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    def control_door1_open2_left(self):
        # 开屏柜门 左拉 正面
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        if self.arrival_flag == 0:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.15, self.current_request.offset[1] + 0.035)
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2] + 0.103
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.1
            move0 = moveJ_P_TargetPose
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        
        self.set_gripper(600)
        rospy.sleep(2.0)

        if self.arrival_flag == 1:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.15, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.close_gripper()
        rospy.sleep()

        if self.arrival_flag == 2:
            l = 0.64
            theta = np.pi/8
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            dx, dy = self.dxy(-l*np.sin(theta), l*(1-np.cos(theta)))
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movej_P_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 3:
            theta = np.pi/4
            next_pose = copy.deepcopy(current_pose)
            dx, dy = self.dxy(-l*np.sin(theta), l*(1-np.cos(theta)))
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movej_P_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # if self.arrival_flag == 4:
        #     moveL_TargetPose.Pose.position.z +=  0.05
        #     self.moveL_pub.publish(moveL_TargetPose)
        # while self.arrival_flag == 4 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == 5:
        #     moveL_TargetPose = moveJ_P_TargetPose
        #     dx, dy = self.dxy(0.02, 0)
        #     moveL_TargetPose.Pose.position.x +=  dx
        #     moveL_TargetPose.Pose.position.y +=  dy
        #     self.moveL_pub.publish(moveL_TargetPose)
        # while self.arrival_flag == 5 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == 6:
        #     dx, dy = self.dxy(0, 0.07)
        #     moveL_TargetPose.Pose.position.x +=  dx
        #     moveL_TargetPose.Pose.position.y +=  dy
        #     self.moveL_pub.publish(moveL_TargetPose)
        # while self.arrival_flag == 6 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == 7:
        #     moveL_TargetPose.Pose.position.z +=  0.025
        #     self.moveL_pub.publish(moveL_TargetPose)
        # while self.arrival_flag == 7 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == 8:
        #     dx, dy = self.dxy(-0.1, 0)
        #     moveL_TargetPose.Pose.position.x +=  dx
        #     moveL_TargetPose.Pose.position.y +=  dy
        #     self.moveL_pub.publish(moveL_TargetPose)
        # while self.arrival_flag == 8 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == 9:
        #     moveL_TargetPose.Pose.position.z +=  -0.1
        #     self.moveL_pub.publish(moveL_TargetPose)
        # while self.arrival_flag == 9 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == 10:
        #     dx, dy = self.dxy(-0.1, -0.2)
        #     moveL_TargetPose.Pose.position.x +=  dx
        #     moveL_TargetPose.Pose.position.y +=  dy
        #     self.moveL_pub.publish(moveL_TargetPose)
        # while self.arrival_flag == 10  and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------屏柜门正面关-------------------------------------------
    def control_door1_close_right(self):
        # 关右屏柜门 正面 第1步
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0
        if self.arrival_flag == i:
            joint = [0, -82, 10, 90, 0, -90]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
      
        if self.arrival_flag == i:
            joint = [16, -82, 10, 90, -45, -80]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.03
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # if self.arrival_flag == i:
        #     joint = [16, -82, 40, 90, -45, -56]
        #     joint_rad = [math.radians(angle) for angle in joint]
        #     moveJ_TargetPose.joint = joint_rad
        #     moveJ_TargetPose.speed = 0.03
        #     self.moveJ_pub.publish(moveJ_TargetPose)
        #     i +=1
        # while self.arrival_flag == i - 1 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == i:
        #     joint = [6, -82, 40, 90, -95, -46]
        #     joint_rad = [math.radians(angle) for angle in joint]
        #     moveJ_TargetPose.joint = joint_rad
        #     moveJ_TargetPose.speed = 0.03
        #     self.moveJ_pub.publish(moveJ_TargetPose)
        #     i += 1
        # while self.arrival_flag == i - 1 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == i:
        #     joint = [12, -82, 40, 90, -103, -46]
        #     joint_rad = [math.radians(angle) for angle in joint]
        #     moveJ_TargetPose.joint = joint_rad
        #     moveJ_TargetPose.speed = 0.05
        #     self.moveJ_pub.publish(moveJ_TargetPose)
        #     i += 1
        # while self.arrival_flag == i - 1 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == i:
        #     joint = [36, -88, 70, 95, -97, -23]
        #     joint_rad = [math.radians(angle) for angle in joint]
        #     moveJ_TargetPose.joint = joint_rad
        #     moveJ_TargetPose.speed = 0.05
        #     self.moveJ_pub.publish(moveJ_TargetPose)
        #     i += 1
        # while self.arrival_flag == i - 1 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == i:
        #     joint = [74, -98, 61, 52, -67, 20]
        #     joint_rad = [math.radians(angle) for angle in joint]
        #     moveJ_TargetPose.joint = joint_rad
        #     moveJ_TargetPose.speed = 0.05
        #     self.moveJ_pub.publish(moveJ_TargetPose)
        #     i += 1
        # while self.arrival_flag == i - 1 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)



        if self.arrival_flag == i:
            joint = [16, -27, -70, 90, -100, -100]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.03
            self.moveJ_pub.publish(moveJ_TargetPose)
            i +=1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            joint = [70, -10, -92, 90, -68, -100]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.03
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            joint = [86, -30, -68, 100, -45, -50]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.05
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)





        if self.arrival_flag == i:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            next_pose[2] += 0.03
            self.robot.Movel_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.14, self.current_request.offset[1]+0.01)
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            next_pose[0] = self.target_pose.pose[0] + dx
            next_pose[1] = self.target_pose.pose[1] + dy
            next_pose[2] = self.target_pose.pose[2] + self.current_request.offset[2] + 0.17
            next_pose[3] = 1.57
            next_pose[4] = -1.57
            next_pose[5] = 0
            self.robot.Movej_P_Cmd(next_pose, 5, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += -np.pi/2
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.get_target_pose(0, 0.015, 0)
        self.arrival_flag = i

        if self.arrival_flag == i:
            dx, dy = self.dxy(0.023, self.current_request.offset[1]-0.03)
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            next_pose[0] = self.target_pose.pose[0] + dx
            next_pose[1] = self.target_pose.pose[1] + dy
            next_pose[2] = self.target_pose.pose[2] + self.current_request.offset[2] + 0.18
            next_pose[3] = -1.5
            next_pose[4] = -1
            next_pose[5] = -2.8
            self.robot.Movej_P_Cmd(next_pose, 5, 0, trajectory_connect=0, block=True)
            i += 1
            print(f"okok")
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            next_pose[2] += -0.18
            self.robot.Movel_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.015, -0.05)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movel_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.18, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movel_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    def control_door1_close2_right(self):
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0
        if self.arrival_flag == i:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.09, self.current_request.offset[1] - 0.078)
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2] + 0.07
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        
        if self.arrival_flag == i:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.07, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, 0.063)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, -0.02)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.07, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, 0.038)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0.06, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            moveL_TargetPose.Pose.position.z += -0.045
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0.045, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.15, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    # def control_door1_close1_right(self):
    #     # 关右屏柜门 正面 第1步
    #     moveJ_TargetPose = MoveJ()
    #     moveJ_P_TargetPose = MoveJ_P()
    #     moveL_TargetPose = MoveL()
    #     if self.arrival_flag == 0:
    #         joint = [0, -75, -10, -90, 0, 90]
    #         joint_rad = [math.radians(angle) for angle in joint]
    #         moveJ_TargetPose.joint = joint_rad
    #         moveJ_TargetPose.speed = 0.1
    #         self.moveJ_pub.publish(moveJ_TargetPose)
    #     while self.arrival_flag == 0 and not rospy.is_shutdown():
    #         pass
    #     rospy.sleep(0.6)
      
    #     if self.arrival_flag == 1:
    #         joint = [9.282, -74.998, 0.008, -90.012, 30.628, 89.99]
    #         joint_rad = [math.radians(angle) for angle in joint]
    #         moveJ_TargetPose.joint = joint_rad
    #         moveJ_TargetPose.speed = 0.1
    #         self.moveJ_pub.publish(moveJ_TargetPose)
    #     while self.arrival_flag == 1 and not rospy.is_shutdown():
    #         pass
    #     rospy.sleep(0.6)

    #     if self.arrival_flag == 2:
    #         joint = [0.236, -75.009, 10.943, -89.007, 81.756, 110.116]
    #         joint_rad = [math.radians(angle) for angle in joint]
    #         moveJ_TargetPose.joint = joint_rad
    #         moveJ_TargetPose.speed = 0.1
    #         self.moveJ_pub.publish(moveJ_TargetPose)
    #     while self.arrival_flag == 2 and not rospy.is_shutdown():
    #         pass
    #     rospy.sleep(0.6)

    #     if self.arrival_flag == 3:
    #         joint = [9.082, -76.359, 67.943, -106.990, 85.870, 166.385]
    #         joint_rad = [math.radians(angle) for angle in joint]
    #         moveJ_TargetPose.joint = joint_rad
    #         moveJ_TargetPose.speed = 0.1
    #         self.moveJ_pub.publish(moveJ_TargetPose)
    #     while self.arrival_flag == 3 and not rospy.is_shutdown():
    #         pass
    #     rospy.sleep(0.6)

    #     if self.arrival_flag == 4:
    #         joint = [53.508, -90.461, 94.614, -129.968, 86.822, 174.807]
    #         joint_rad = [math.radians(angle) for angle in joint]
    #         moveJ_TargetPose.joint = joint_rad
    #         moveJ_TargetPose.speed = 0.1
    #         self.moveJ_pub.publish(moveJ_TargetPose)
    #     while self.arrival_flag == 4 and not rospy.is_shutdown():
    #         pass
    #     rospy.sleep(0.6)

    #     if self.arrival_flag == 5:
    #         joint = [77.669, -95.244, 73.089, -156.525, 74.382, 163.563]
    #         joint_rad = [math.radians(angle) for angle in joint]
    #         moveJ_TargetPose.joint = joint_rad
    #         moveJ_TargetPose.speed = 0.1
    #         self.moveJ_pub.publish(moveJ_TargetPose)
    #     while self.arrival_flag == 5 and not rospy.is_shutdown():
    #         pass
    #     rospy.sleep(0.6)

    #     if self.arrival_flag == 6:
    #         joint = [81.735, -104.775, 74.597, -165.629, 61.258, 170.808]
    #         joint_rad = [math.radians(angle) for angle in joint]
    #         moveJ_TargetPose.joint = joint_rad
    #         moveJ_TargetPose.speed = 0.1
    #         self.moveJ_pub.publish(moveJ_TargetPose)
    #     while self.arrival_flag == 6 and not rospy.is_shutdown():
    #         pass
    #     rospy.sleep(1.0)

    #     # if self.arrival_flag == 5:
    #     #     self.get_state()
    #     #     dx, dy = self.dxy(0.008, 0)
    #     #     moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
    #     #     moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
    #     #     moveL_TargetPose.Pose.position.z = self.current_pose[2]
    #     #     moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
    #     #     moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
    #     #     moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
    #     #     moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
    #     #     moveL_TargetPose.trajectory_connect = 0
    #     #     moveL_TargetPose.speed = 0.15
    #     #     self.moveL_pub.publish(moveL_TargetPose)
    #     # while self.arrival_flag == 5 and not rospy.is_shutdown():
    #     #     pass
    #     # rospy.sleep(1.0)

    #     # if self.arrival_flag == 6:
    #     #     dx, dy = self.dxy(0.0015, 0)
    #     #     moveL_TargetPose.Pose.position.x += dx
    #     #     moveL_TargetPose.Pose.position.y += dy
    #     #     moveL_TargetPose.Pose.position.z += -0.078
    #     #     moveL_TargetPose.speed = 0.15
    #     #     self.moveL_pub.publish(moveL_TargetPose)
    #     # while self.arrival_flag == 6 and not rospy.is_shutdown():
    #     #     pass
    #     # rospy.sleep(1.0)

    #     # if self.arrival_flag == 7:
    #     #     dx, dy = self.dxy(0.0015, 0)
    #     #     moveL_TargetPose.Pose.position.x += dx
    #     #     moveL_TargetPose.Pose.position.y += dy
    #     #     moveL_TargetPose.speed = 0.15
    #     #     self.moveL_pub.publish(moveL_TargetPose)
    #     # while self.arrival_flag == 7 and not rospy.is_shutdown():
    #     #     pass
    #     # rospy.sleep(1.0)

    #     # if self.arrival_flag == 8:
    #     #     dx, dy = self.dxy(0.001, -0.02)
    #     #     moveL_TargetPose.Pose.position.x += dx
    #     #     moveL_TargetPose.Pose.position.y += dy
    #     #     moveL_TargetPose.speed = 0.15
    #     #     self.moveL_pub.publish(moveL_TargetPose)
    #     # while self.arrival_flag == 8 and not rospy.is_shutdown():
    #     #     pass
    #     # rospy.sleep(1.0)

    #     # if self.arrival_flag == 9:
    #     #     dx, dy = self.dxy(-0.1, 0)
    #     #     moveL_TargetPose.Pose.position.x += dx
    #     #     moveL_TargetPose.Pose.position.y += dy
    #     #     moveL_TargetPose.speed = 0.15
    #     #     self.moveL_pub.publish(moveL_TargetPose)
    #     # while self.arrival_flag == 9 and not rospy.is_shutdown():
    #     #     pass
    #     # rospy.sleep(1.0)

    #     self.arrival_flag = 0
    #     print("操作完成")

    def control_door1_close1_left(self):
        # 关左屏柜门 正面 第1步
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        if self.arrival_flag == 0:
            moveJ_TargetPose.joint = [0, -0.523599, -1.047198, 1.570796, 0, -1.570796]
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.6)
      
        if self.arrival_flag == 1:
            joint = [14.908, -29.985, -59.993, 89.962, 81.445, -90.569]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.6)

        if self.arrival_flag == 2:
            joint = [-15.534, 13.141, -101.256, 89.968, 74.562, -90.565]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.6)

        if self.arrival_flag == 3:
            joint = [-54.338, -2.061, -91.407, 90.078, 32.97, -90.55]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.6)

        if self.arrival_flag == 4:
            joint = [-59.878, -11.541, -86.065, 89.075, 24.921, -90.492]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 4 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 5:
            self.get_state()
            dx, dy = self.dxy(0.008, 0)
            moveL_TargetPose.Pose.position.x = self.current_pose[0] + dx
            moveL_TargetPose.Pose.position.y = self.current_pose[1] + dy
            moveL_TargetPose.Pose.position.z = self.current_pose[2]
            moveL_TargetPose.Pose.orientation.x = self.current_pose[3]
            moveL_TargetPose.Pose.orientation.y = self.current_pose[4]
            moveL_TargetPose.Pose.orientation.z = self.current_pose[5]
            moveL_TargetPose.Pose.orientation.w = self.current_pose[6]
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 5 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 6:
            dx, dy = self.dxy(0.0015, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.Pose.position.z += -0.078
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 6 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 7:
            dx, dy = self.dxy(0.0015, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 7 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 8:
            dx, dy = self.dxy(0.001, -0.02)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 8 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 9:
            dx, dy = self.dxy(-0.1, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 9 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    def control_door1_close2(self):
        # 关屏柜门 正面 第2步
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        if self.arrival_flag == 0:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.12, self.current_request.offset[1] - 0.01) 
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2] + 0.02
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.1
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 1:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.08, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            moveL_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 2:
            moveL_TargetPose.Pose.position.z +=  -0.02
            moveL_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 3:
            dx, dy = self.dxy(0.045, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            moveL_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 4:
            dx, dy = self.dxy(-0.1, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            moveL_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 4 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------屏柜门背面开-------------------------------------------
    def control_door2_open2_0(self, l=0.275):
        # 开屏柜门 背面
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        if self.arrival_flag == 0:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1]) 
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2]
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            move0 = moveJ_P_TargetPose
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        
        # 按开关
        if self.arrival_flag == 1:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.072, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 2:
            dx, dy = self.dxy(-0.05, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # 转把手
        if self.arrival_flag == 3:
            dx, dy = self.dxy(0, 0.03)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 4:
            moveL_TargetPose.Pose.position.z +=  0.05
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 4 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 5:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.02, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 5 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 6:
            dx, dy = self.dxy(0, -0.07)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 6 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 7:
            moveL_TargetPose.Pose.position.z +=  0.034        #0.039
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 7 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 8:
            dx, dy = self.dxy(-0.07, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 8 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # # 抓把手
        # if self.arrival_flag == 9:
        #     dx, dy = self.dxy(0, -0.018)
        #     moveL_TargetPose.Pose.position.x +=  dx
        #     moveL_TargetPose.Pose.position.y +=  dy
        #     self.moveL_pub.publish(moveL_TargetPose)
        # while self.arrival_flag == 9 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == 10:
        #     moveL_TargetPose.Pose.position.z +=  0.0128
        #     self.moveL_pub.publish(moveL_TargetPose)
        # while self.arrival_flag == 10 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # self.set_gripper(450)
        # rospy.sleep(2.0)

        # if self.arrival_flag == 11:
        #     dx, dy = self.dxy(0.065, 0)
        #     moveL_TargetPose.Pose.position.x +=  dx
        #     moveL_TargetPose.Pose.position.y +=  dy
        #     self.moveL_pub.publish(moveL_TargetPose)
        # while self.arrival_flag == 11 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # self.close_gripper(500)
        # rospy.sleep(2.0)

        # # 拉把手
        # if self.arrival_flag == 12:
        #     s=np.sqrt(0.04**2+l**2)
        #     alpha = np.arctan(0.04/l)
        #     theta = np.pi/8
        #     dx, dy = self.dxy(-s*np.sin(alpha+theta)+0.04, -s*(np.cos(alpha)-np.cos(alpha+theta)))
        #     moveJ_P_TargetPose = copy.deepcopy(moveL_TargetPose)
        #     print(f"1.{moveL_TargetPose}")
        #     moveJ_P_TargetPose.Pose.position.x +=  dx
        #     moveJ_P_TargetPose.Pose.position.y +=  dy
        #     print(f"{dx},{dy}")
        #     print(f"{moveJ_P_TargetPose}")
        #     moveJ_P_TargetPose.speed = 0.1
        #     self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        # while self.arrival_flag == 12 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # if self.arrival_flag == 13:
        #     theta = np.pi/4
        #     dx, dy = self.dxy(-s*np.sin(alpha+theta)+0.04, -s*(np.cos(alpha)-np.cos(alpha+theta)))
        #     moveJ_P_TargetPose = moveL_TargetPose
        #     print(f"2.{moveL_TargetPose}")
        #     moveJ_P_TargetPose.Pose.position.x +=  dx
        #     moveJ_P_TargetPose.Pose.position.y +=  dy
        #     print(f"{dx},{dy}")
        #     print(f"{moveJ_P_TargetPose}")
        #     moveJ_P_TargetPose.speed = 0.1
        #     self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        # while self.arrival_flag == 13 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        # self.set_gripper(450)
        # rospy.sleep(2.0)

        self.arrival_flag = 0
        print("操作完成")

    def control_door2_open(self, l=0.25):
        # 开屏柜门 背面
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        if self.arrival_flag == 0:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1]) 
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2]
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            move0 = moveJ_P_TargetPose
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        
        # 按开关
        if self.arrival_flag == 1:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.072, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 2:
            dx, dy = self.dxy(-0.05, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # 转把手
        if self.arrival_flag == 3:
            dx, dy = self.dxy(0, 0.03)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 4:
            moveL_TargetPose.Pose.position.z +=  0.05
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 4 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 5:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.02, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 5 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 6:
            dx, dy = self.dxy(0, -0.07)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 6 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 7:
            moveL_TargetPose.Pose.position.z +=  0.039
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 7 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 8:
            dx, dy = self.dxy(-0.07, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 8 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # 抓把手
        if self.arrival_flag == 9:
            dx, dy = self.dxy(0, -0.018)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 9 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 10:
            moveL_TargetPose.Pose.position.z +=  0.0128
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 10 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper(450)
        rospy.sleep(2.0)

        if self.arrival_flag == 11:
            dx, dy = self.dxy(0.065, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 11 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.close_gripper(800)
        rospy.sleep(2.0)

        # 拉把手
        if self.arrival_flag == 12:
            s=np.sqrt(0.06**2+l**2)
            theta = np.arctan(6/25)
            dx, dy = self.dxy(-s*np.sin(np.pi/8+theta)+0.06, s*np.cos(np.pi/8+theta)-0.25)
            moveJ_P_TargetPose = moveL_TargetPose
            moveJ_P_TargetPose.Pose.position.x +=  dx
            moveJ_P_TargetPose.Pose.position.y +=  dy
            moveJ_P_TargetPose.speed = 0.1
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 12 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper(450)
        rospy.sleep(2.0)

        if self.arrival_flag == 13:
            dx, dy = self.dxy(-0.05, 0)
            moveL_TargetPose = moveJ_P_TargetPose
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 13 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper(0)
        rospy.sleep(2.0)

        if self.arrival_flag == 14:
            dx, dy = self.dxy(0, 0.16)
            moveJ_P_TargetPose = moveJ_P_TargetPose
            moveJ_P_TargetPose.Pose.position.x +=  dx
            moveJ_P_TargetPose.Pose.position.y +=  dy
            moveJ_P_TargetPose.Pose.position.z +=  0.05
            moveJ_P_TargetPose.Pose.orientation.x = -0.575688
            moveJ_P_TargetPose.Pose.orientation.y = -0.410942
            moveJ_P_TargetPose.Pose.orientation.z = -0.574655
            moveJ_P_TargetPose.Pose.orientation.w = 0.411681
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 14 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 15:
            dx, dy = self.dxy(0.165, 0)
            moveL_TargetPose = moveJ_P_TargetPose
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 15 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 16:
            dx, dy = self.dxy(0, -0.04)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 16 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 17:
            dx, dy = self.dxy(0.03, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 17 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 18:
            dx, dy = self.dxy(0, -0.045)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 18 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 19:
            dx, dy = self.dxy(-0.08, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 19 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 20:
            dx, dy = self.dxy(0, -0.06)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 20 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 21:
            dx, dy = self.dxy(-0.05, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 21 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 22:
            moveJ_TargetPose = MoveJ()
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[0] += -np.pi/8
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 22 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 23:
            dx, dy = self.dxy(0.05, 0.05)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            moveL_TargetPose.Pose.orientation.x = -0.411223
            moveL_TargetPose.Pose.orientation.y = -0.574982
            moveL_TargetPose.Pose.orientation.z = -0.410485
            moveL_TargetPose.Pose.orientation.w = 0.576016
            moveJ_TargetPose.speed = 0.06
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 23 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 24:
            dx, dy = self.dxy(0.17, 0)
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            moveL_TargetPose.Pose.orientation.x = -0.339140
            moveL_TargetPose.Pose.orientation.y = -0.620297
            moveL_TargetPose.Pose.orientation.z = -0.338870
            moveL_TargetPose.Pose.orientation.w = 0.620792
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 24 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 25:
            dx, dy = self.dxy(0, 0.09) 
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            moveJ_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 25 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 26:
            dx, dy = self.dxy(-0.1, 0) 
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            moveJ_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 26 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 27:
            dx, dy = self.dxy(0, 0.05) 
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 27 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 28:
            dx, dy = self.dxy(-0.05, 0.03) 
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 28 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 29:
            dx, dy = self.dxy(-0.1, 0) 
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 29 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 30:
            dx, dy = self.dxy(0, 0.05) 
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 30 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 31:
            dx, dy = self.dxy(-0.05, 0) 
            moveL_TargetPose.Pose.position.x +=  dx
            moveL_TargetPose.Pose.position.y +=  dy
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 31 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 32:
            moveJ_TargetPose = MoveJ()
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[0] += np.pi/8
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 32 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    def control_door3_open(self, l=0.4):
        # 开防火门
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0
        if self.arrival_flag == i:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1])
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2]
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.15
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
        
        self.set_gripper(500)
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            moveL_TargetPose = moveJ_P_TargetPose
            dx, dy = self.dxy(0.08, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            self.moveL_pub.publish(moveL_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper(0)
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            theta = np.pi/5
            dx, dy = self.dxy(0, 0.08*(1-np.cos(theta)))
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += -0.08*np.sin(theta)
            next_pose[3] = 1.57
            next_pose[4] = -1.57 - theta/2
            next_pose[5] = 0
            self.robot.Movej_P_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            next_pose = copy.deepcopy(current_pose)
            theta = np.pi/2.5
            dx, dy = self.dxy(0, 0.07*(1-np.cos(theta)))
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += -0.07*np.sin(theta)
            next_pose[3] = 1.57
            next_pose[4] = -1.57 - theta/2
            next_pose[5] = 0
            self.robot.Movej_P_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            s=np.sqrt(0.035**2+l**2)
            alpha = np.arctan(0.035/l)
            theta = np.pi/8
            dx, dy = self.dxy(-s*np.sin(alpha+theta)+0.055, -s*(np.cos(alpha)-np.cos(alpha+theta)))
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[5] += theta
            self.robot.Movej_P_Cmd(next_pose, 5, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            next_pose = copy.deepcopy(current_pose)
            theta = np.pi/4
            dx, dy = self.dxy(-s*np.sin(alpha+theta)+0.055, -s*(np.cos(alpha)-np.cos(alpha+theta)))
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[5] += theta/2
            self.robot.Movej_P_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        self.set_gripper(800)
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += 0.07
            self.robot.Movej_P_Cmd(next_pose, 5, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.15, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movej_P_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
           pass
        rospy.sleep(1.0)

        self.set_gripper(0)
        rospy.sleep(2.0)

        if self.arrival_flag == i:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            dx, dy = self.dxy(0, 0.18)
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += 0
            next_pose[3] = 1.57
            next_pose[4] = -1.57
            next_pose[5] = 0
            self.robot.Movel_Cmd(next_pose, 8, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            dx, dy = self.dxy(0, 0.03)
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += 0.18
            next_pose[3] = 2.4
            next_pose[4] = -1.45
            next_pose[5] = -1.95
            self.robot.Movej_P_Cmd(next_pose, 8, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i-1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0.2, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movej_P_Cmd(next_pose, 8, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.08, -0.55)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movel_Cmd(next_pose, 15, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, 0.2)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movel_Cmd(next_pose, 15, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0.4, 0.3)
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += -0.18
            next_pose[3] = 1.75
            next_pose[4] = -1.51
            next_pose[5] = 0.344
            self.robot.Movej_P_Cmd(next_pose, 15, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0, 0.09)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movel_Cmd(next_pose, 15, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            self.get_state()
            rospy.sleep(0.5)
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[4] += -np.pi/5
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            dx, dy = self.dxy(0, 0.04)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movej_P_Cmd(next_pose, 8, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i-1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            self.get_state()
            rospy.sleep(0.5)
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[4] += -np.pi/5
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[0] += np.pi/6
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            dx, dy = self.dxy(0, -0.2)
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += 0
            next_pose[3] = 1.75
            next_pose[4] = -1.51
            next_pose[5] = 0.34
            self.robot.Movej_P_Cmd(next_pose, 8, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i-1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.set_gripper(0)
        rospy.sleep(2.0)

        self.arrival_flag = 0
        print("操作完成")

    def control_door3_close(self):
        # 关右屏柜门 正面 第1步
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0
        if self.arrival_flag == i:
            joint = [91.410, 31.864, -125.508, 90.044, 91.410, -86.357]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
      
        if self.arrival_flag == i:
            joint = [83.932, -37.606, -59.071, 89.246, 83.978, -83.281]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            dx, dy = self.dxy(0.18, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += 0
            self.robot.Movel_Cmd(next_pose, 8, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    def control_door3_close2(self):
        # 关右屏柜门 正面 第1步
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0
        if self.arrival_flag == i:
            self.get_state()
            rospy.sleep(0.5)
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[4] += np.pi/5
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
      
        if self.arrival_flag == i:
            moveJ_TargetPose.joint[0] += -np.pi/4.5
            moveJ_TargetPose.joint[4] += -np.pi/7
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        # if self.arrival_flag == i:
        #     dx, dy = self.dxy(-0.1, 0)
        #     current_pose = self.robot.Get_Current_Arm_State()[2]
        #     next_pose = copy.deepcopy(current_pose)
        #     next_pose[0] += dx
        #     next_pose[1] += dy
        #     self.robot.Movel_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
        #     i += 1
        # while self.arrival_flag == i - 1 and not rospy.is_shutdown():
        #     pass
        # rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.05, 0)
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += 0
            next_pose[3] = 0
            next_pose[4] = -1.57
            next_pose[5] = 0
            self.robot.Movel_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(self.current_request.offset[0]-0.05, self.current_request.offset[1])
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            next_pose[0] = self.target_pose.pose[0] + dx
            self.robot.Movel_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.1, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movel_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    def control_door3_close3(self):
        # 关右屏柜门 正面 第1步
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0
        if self.arrival_flag == i:
            joint = [-102.415, 19.639, -124.342, 86.847, -102.014, -105.031]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
      
        if self.arrival_flag == i:
            joint = [-87.74, -29.267, -67.636, 97.194, -86.467, -100.38]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            dx, dy = self.dxy(0.15, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += 0
            self.robot.Movel_Cmd(next_pose, 8, 0, trajectory_connect=0, block=True)      
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            self.get_state()
            rospy.sleep(0.5)
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[4] += -np.pi/5
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i -1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)
      
        if self.arrival_flag == i:
            moveJ_TargetPose.joint[0] += np.pi/4.5
            moveJ_TargetPose.joint[4] += np.pi/7
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.1, 0.07)
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            next_pose[0] += dx
            next_pose[1] += dy
            next_pose[2] += 0
            next_pose[3] = 0
            next_pose[4] = -1.57
            next_pose[5] = 0
            self.robot.Movel_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(self.current_request.offset[0]+0.01, self.current_request.offset[1])
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            next_pose[0] = self.target_pose.pose[0] + dx
            self.robot.Movej_P_Cmd(next_pose, 15, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.1, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movel_Cmd(next_pose, 10, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

if __name__ == '__main__':
    try:
        arm = arm_control()
        arm.run()
        # 保持节点运行
        rospy.spin()
    except rospy.ROSInterruptException:
        pass