#!/usr/bin/env python3
from e2q import euler_to_quaternion
import math
import copy
import rospy
import threading
import numpy as np
from exception import RobotArmMoveException
from robotic_arm import *
from std_msgs.msg import Empty, Bool
from njust.msg import result
from njust.srv import task, taskResponse, pose, poseResponse, poseRequest
from rm_msgs.msg import ArmState, Arm_Current_State, Gripper_Pick, Gripper_Set, GetArmState_Command, MoveJ, MoveJ_P, \
    MoveL, Plan_State
from rm_msgs.msg import Six_Force, Pos_Teach, Stop_Teach


class arm_control:
    def __init__(self):
        rospy.init_node('arm_control')
        rospy.sleep(1.0)

        # 存储参数
        self.current_joint = [0, 0, 0, 0, 0, 0]
        self.current_pose = [0, 0, 0, 0, 0, 0, 0]
        self.current_pose_n = [0, 0, 0, 0, 0, 0]
        self.target_pose = [0, 0, 0, 0, 0, 0, 0]
        self.target_pose_1 = [0, 0, 0, 0, 0, 0, 0]
        self.dzx_1 = [0, 0, 0]
        self.current_request = None
        self.action = False
        self.arm_dof = 6
        self.arrival_flag = 0
        self.pose_flag = False
        self.get = True
        self.plan_num = 0
        self.clear_result_received = False
        self.clear_success = False

        ip_address = "192.168.1.18"
        model = RML63_II
        self.robot = Arm(model, ip_address)  # 直接传递字符串，不需要转换
        if self.robot.API_Version():
            print(f"成功连接到机械臂IP地址{ip_address}")
        else:
            print("连接机械臂失败。")

        # 六维力传感器相关
        # self.six_force_sub = rospy.Subscriber("/rm_driver/UdpSixZeroForce", Six_Force, self.six_force_callback)
        self.clear_force_data_pub = rospy.Publisher("/rm_driver/ClearForceData_Cmd", Empty, queue_size=10)
        self.clear_result_sub = rospy.Subscriber("/rm_driver/ClearForceData_Result", Bool, self.clear_result_callback)
        self.arm_position_teach_pub = rospy.Publisher("/rm_driver/Arm_PosTeach", Pos_Teach, queue_size=10)
        self.arm_teach_stop_pub = rospy.Publisher("/rm_driver/Arm_StopTeach", Stop_Teach, queue_size=10)
        # 订阅机械臂状态
        self.Arm_Base_State_sub = rospy.Subscriber("/rm_driver/ArmCurrentState", ArmState, self.Get_Arm_State_Callback)
        self.Arm_New_State_sub = rospy.Subscriber("/rm_driver/Arm_Current_State", Arm_Current_State,
                                                  self.GetArmState_Callback)
        # 发布夹爪控制命令
        self.gripper_pub = rospy.Publisher('/rm_driver/Gripper_Pick', Gripper_Pick, queue_size=10)
        # 发布夹爪到设定位置命令
        self.gripper_pose_pub = rospy.Publisher('/rm_driver/Gripper_Set', Gripper_Set, queue_size=10)

        self.stop_pub = rospy.Publisher('/rm_driver/Emergency_Stop', Empty, queue_size=10)

        self.moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=10)
        self.moveJ_P_pub = rospy.Publisher("/rm_driver/MoveJ_P_Cmd", MoveJ_P, queue_size=10)
        self.moveL_pub = rospy.Publisher("/rm_driver/MoveL_Cmd", MoveL, queue_size=10)

        self.result_pub = rospy.Publisher("/result_info", result, queue_size=10)

        rospy.sleep(0.5)

        self.planState_sub = rospy.Subscriber("/rm_driver/Plan_State", Plan_State, self.planStateCallback)

        self.command = GetArmState_Command()
        self.command.command = "get_current_arm_state"
        self.test_Get_Arm_Base_State_pub = rospy.Publisher("/rm_driver/GetArmState_Cmd", GetArmState_Command,
                                                           queue_size=10)

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
                    if self.current_request.type == 'ts':
                        # 测试
                        self.position_teach('z', 'pos', threshold=5.0, v=5)
                        self.result_pub.publish("success")

                    if self.current_request.type == 'd':
                        # 转方向
                        if self.current_request.direction == 20:
                            self.close_gripper()
                            rospy.sleep(1.5)
                        self.init_position(self.current_request.direction)
                        self.close_gripper()
                        rospy.sleep(1.5)
                        self.result_pub.publish("success")

                    if self.current_request.type == 'p':
                        #
                        self.get_target_pose()
                        self.result_pub.publish("success")

                    if self.current_request.type == 'press':
                        self.init_position(self.current_request.direction)
                        self.position_dxyz(0.1, 0, -0.15)
                        self.get_target_pose()
                        if self.check_target_found():
                            continue
                        self.press()
                        self.result_pub.publish("success")

                    if self.current_request.type == 'button':
                        # 按钮
                        self.init_position(self.current_request.direction + 10)
                        self.position_dxyz(0.35, -0.08, 0, 1)
                        self.get_target_pose()
                        if self.target_pose.pose[0] > 2:
                            print(f"failure: target not found")
                            self.position_dxyz(0.05, 0.03, 0, 2)
                        if self.check_target_found(10):
                            continue
                        self.control_button_q()
                        self.init_position(self.current_request.direction + 10)
                        self.result_pub.publish("success")

                    if self.current_request.type == 'switch':
                        self.init_position(self.current_request.direction + 10)
                        if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                            # 压板投入
                            dy = (self.current_request.rank[0]+self.current_request.rank[1]) / 100
                            self.position_dxyz(0.3, -0.12-dy, -0.69, 1)
                            self.position_dxyz(0.1, 0, 0, 2)
                            self.delta_joint([0,0,0,0,0,-3.8])
                            self.get_target_pose(-2)
                            if self.check_target_found(10):
                                continue
                            self.control_switch_open(0.033)

                            self.get_target_pose(-2)
                            if self.check_target_found(10):
                                continue

                            self.control_switch_open2(0.0305)
                            self.init_position(self.current_request.direction + 10)
                            self.result_pub.publish("success")
                        if self.current_request.now_state == 1 and self.current_request.next_state == 0:
                            # 压板退出
                            self.position_dxyz(0.3, -0.15, -0.69, 1)
                            self.position_dxyz(0.1, 0, 0, 2)
                            self.delta_joint([0,0,0,0,0,-3])
                            self.get_target_pose(-2)
                            if self.check_target_found(10):
                                continue
                            self.control_switch_close()
                            self.init_position(self.current_request.direction + 10)
                            self.result_pub.publish("success")

                    if self.current_request.type == 'pos':
                        if self.current_request.now_state == 0 and self.current_request.next_state == 0:
                            self.init_position(1)
                            self.position_dxyz(0.2, 0, 0.1)
                            self.delta_joint([-18,0,0,0,0,0], v=0.06)
                            self.delta_joint([8,0,0,0,0,0], v=0.1)
                            self.result_pub.publish("success")
                        if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                            self.position_dxyz(0, 0, -0.2)
                            self.result_pub.publish("success")

                    if self.current_request.type == 'door1':
                        if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                            # 柜门正面 开
                            self.init_position(self.current_request.direction + 10)
                            self.position_dxyz(0.3, 0, -0.2, 1)
                            self.get_target_pose()
                            if self.check_target_found(10):
                                continue
                            self.control_door1_open_right_q()
                            self.init_position(self.current_request.direction + 10)
                            self.result_pub.publish("success")
                        if self.current_request.now_state == 1 and self.current_request.next_state == 0:
                            # 柜门正面 关
                            self.init_position(self.current_request.direction + 10)
                            if self.current_request.direction == 1:
                                self.position_dxyz(0.3, 0, -0.2, 1)
                                self.get_target_pose()
                                if self.check_target_found(10):
                                    continue
                                self.init_position(self.current_request.direction + 10)
                                self.init_position(12)

                                self.control_door1_close_right_q4()

                                self.get_target_pose()
                                if self.check_target_found(10):
                                    continue
                                self.control_door1_close2_right()
                                self.init_position(self.current_request.direction + 10)
                                self.init_position(0)
                                self.result_pub.publish("success")

                    if self.current_request.type == 'door3':
                        # 淮安 防火门
                        if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                            self.init_position(12)
                            self.position_dxyz(0.39, -0.08, -0.25)
                            self.get_target_pose()
                            if self.check_target_found(10):
                                continue
                            self.control_door3_open_n()
                            self.init_position(12)
                            self.result_pub.publish("success")
                        if self.current_request.now_state == 1 and self.current_request.next_state == 0:
                            self.init_position(12)
                            self.init_position(11)
                            self.control_door3_close()
                            self.get_target_pose(-1)
                            if self.check_target_found(10):
                                continue
                            self.control_door3_close2()
                            self.init_position(12)
                            self.init_position(13)
                            self.control_door3_close3()
                            self.init_position(12)
                            self.result_pub.publish("success")

                    if self.current_request.type == 'door4':
                        if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                            # 左端子箱 开
                            self.init_position(self.current_request.direction)
                            # self.position_dxyz(0.3, -0.1, 0)
                            self.position_dxyz(0.3, 0.05, -0.48)
                            self.get_target_pose()
                            if self.check_target_found():
                                continue
                            
                            self.control_door4_open_1()

                            self.init_position(self.current_request.direction)
                            self.result_pub.publish("success")
                        if self.current_request.now_state == 1 and self.current_request.next_state == 0:
                            # 端子箱 关
                            self.init_position(self.current_request.direction)
                         
                            self.position_dxyz(0.3, 0.05, -0.48)
                            self.get_target_pose()
                            if self.check_target_found():
                                continue
                            self.control_door4_close()
                            self.init_position(self.current_request.direction)
                            self.init_position(0)
                            self.result_pub.publish("success")
                            
                    if self.current_request.type == 'tool':
                        """
                        换装
                        0,0 左退右上  0,1 右退左上
                        1,0 左退     1,1 右退
                        2,0 左上     2,1 右上
                        """
                        self.set_tool_v(0)
                        self.init_position(0)
                        if self.current_request.now_state == 0 and self.current_request.next_state == 0:
                            self.change_tool_left_off()
                            self.init_position(0)
                            self.change_tool_right_on()
                        if self.current_request.now_state == 0 and self.current_request.next_state == 1:
                            self.change_tool_right_off()
                            self.init_position(0)
                            self.change_tool_left_on()
                        if self.current_request.now_state == 1 and self.current_request.next_state == 0:
                            self.change_tool_left_off()
                        if self.current_request.now_state == 1 and self.current_request.next_state == 1:
                            self.change_tool_right_off()
                        if self.current_request.now_state == 2 and self.current_request.next_state == 0:
                            self.change_tool_left_on()
                        if self.current_request.now_state == 2 and self.current_request.next_state == 1:
                            self.change_tool_right_on()
                        self.init_position(0)
                        if self.current_request.now_state == 0 or self.current_request.now_state == 2:
                            self.set_tool_v(1)
                            rospy.sleep(2.0)
                            self.set_tool_v(0)
                            rospy.sleep(2.0)
                            self.set_tool_v(1)
                        if (self.current_request.now_state == 0 and self.current_request.next_state == 0) \
                            or (self.current_request.now_state == 2 and self.current_request.next_state == 1):
                            rospy.sleep(4.0)
                            self.close_gripper()
                            self.close_gripper()
                        rospy.sleep(1.0)
                        self.result_pub.publish("tool_success")

                    self.pose_flag = False
                except Exception as e:
                    rospy.logerr(f"任务执行异常: {e}")
                    # 重置状态
                    self.arrival_flag = 0
                    self.action = False
                finally:
                    self.action = False

            rospy.sleep(0.5)  # 防止CPU占用过高

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
        self.arrival_flag = 0
        self.current_request = req
        print(
            f"收到任务:target={req.target}, type={req.type}, direction={req.direction}, angle={req.angle}, region={req.region}, offset={req.offset}, \
              rank={req.rank}, now_state={req.now_state}, next_state={req.next_state}")
        self.action = True
        if req.type == "stop":
            try:
                self.arrival_flag = -2
                msg = Empty()
                self.stop_pub.publish(msg)
                rospy.loginfo("Emergency stop command sent.")
                self.action = False
                # # 等待并打印结果
                # result = rospy.wait_for_message('/rm_driver/Set_Arm_Stop_Result', Bool, timeout=5.0)
                # rospy.loginfo("Command result: %s", result.data)

            except rospy.ROSException as e:
                rospy.logerr("Failed to get result: %s", str(e))

        return taskResponse("OK")

    def move_jp(self, pose, v, timeout=5):
        """
        pose: [x, y, z, rx, ry, rz]
        v: int
        """
        print(f"move_jp: {pose}")

        # 重试逻辑
        max_retries = 1
        for attempt in range(max_retries):
            try:
                move_jp = MoveJ_P()
                move_jp.Pose.position.x = pose[0]
                move_jp.Pose.position.y = pose[1]
                move_jp.Pose.position.z = pose[2]

                if len(pose) == 6:
                    quaternion = euler_to_quaternion(pose[3], pose[4], pose[5])
                    move_jp.Pose.orientation.x = quaternion[0]
                    move_jp.Pose.orientation.y = quaternion[1]
                    move_jp.Pose.orientation.z = quaternion[2]
                    move_jp.Pose.orientation.w = quaternion[3]
                elif len(pose) == 7:
                    move_jp.Pose.orientation.x = pose[3]
                    move_jp.Pose.orientation.y = pose[4]
                    move_jp.Pose.orientation.z = pose[5]
                    move_jp.Pose.orientation.w = pose[6]

                move_jp.trajectory_connect = 0
                move_jp.speed = v
                self.moveJ_P_pub.publish(move_jp)

                # 等待执行结果
                result = rospy.wait_for_message('/rm_driver/Plan_State', Plan_State, timeout=None)

                if result.state == False:
                    # 规划失败，但不是异常，所以不raise，而是重试
                    rospy.logwarn(f"move_jp 规划失败，第 {attempt + 1} 次重试")

                    # 执行重试操作：先抬升一点，然后调整高度重试
                    if attempt < max_retries - 1:  # 如果不是最后一次尝试
                        self.position_dxyz(0, 0, 0.01, 0, v)  # 抬升 1cm
                        # pose[2] += -2

                        continue  # 继续下一次尝试
                    else:
                        # 最后一次尝试也失败，抛出异常
                        raise RobotArmMoveException("move_jp 规划失败，重试次数用尽")
                else:
                    # 规划成功，重置重试计数器（如果需要）
                    self.plan_num = 0
                    return  # 成功返回

            except rospy.ROSException as e:
                # 等待消息超时
                if attempt < max_retries - 1:
                    rospy.logwarn(f"等待执行结果超时，第 {attempt + 1} 次重试: {e}")
                    rospy.sleep(1)  # 等待1秒后重试
                    continue
                else:
                    raise RobotArmMoveException("move_jp执行结果超时，重试次数用尽")

    def move_l(self, pose, v, timeout=5):
        """
        pose: [x, y, z, rx, ry, rz]
        v: int
        """
        move_l = MoveL()
        print(f"move_l: {pose}")
        move_l.Pose.position.x = pose[0]
        move_l.Pose.position.y = pose[1]
        move_l.Pose.position.z = pose[2]
        if len(pose) == 6:
            quaternion = euler_to_quaternion(pose[3], pose[4], pose[5])
            move_l.Pose.orientation.x = quaternion[0]
            move_l.Pose.orientation.y = quaternion[1]
            move_l.Pose.orientation.z = quaternion[2]
            move_l.Pose.orientation.w = quaternion[3]
        elif len(pose) == 7:
            move_l.Pose.orientation.x = pose[3]
            move_l.Pose.orientation.y = pose[4]
            move_l.Pose.orientation.z = pose[5]
            move_l.Pose.orientation.w = pose[6]
        move_l.trajectory_connect = 0
        move_l.speed = v
        self.moveL_pub.publish(move_l)

        try:
            result = rospy.wait_for_message('/rm_driver/Plan_State', Plan_State, timeout=None)
        except Exception as e:
            print("等待执行结果超时")
            print(e)
            raise RobotArmMoveException("move_l执行结果超时")
        if result.state == False:
            raise RobotArmMoveException("move_l error")

    def move_j(self, joint, v, mod=0, timeout=10):
        move_j = MoveJ()
        if mod == 0:
            print(f"move_j: {joint} a")
            joint_rad = [math.radians(angle) for angle in joint]
            move_j.joint = joint_rad

        if mod == 1:
            print(f"move_j: {joint} r")
            joint_rad = joint
            move_j.joint = joint_rad

        move_j.speed = v
        self.moveJ_pub.publish(move_j)

        try:
            result = rospy.wait_for_message('/rm_driver/Plan_State', Plan_State, timeout=None)
            rospy.sleep(0.1)
        except Exception as e:
            print("等待执行结果超时")
            print(e)
            raise RobotArmMoveException("move_j执行结果超时")
        if result.state == False:
            raise RobotArmMoveException("move_j error")

    def planStateCallback(self, msg):
        if msg.state:
            self.arrival_flag += 1
            print(f"planStateCallback: state = {msg.state}, {self.arrival_flag}")
        else:
            rospy.logerr("Plan State Fail arrival_flag is %d", self.arrival_flag)

    def clear_result_callback(self, msg):
        self.clear_result_received = True
        self.clear_success = msg.data
        print(f"{self.clear_success}")

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

    def GetArmState_Callback(self, msg):
        self.current_pose_n[:] = [msg.Pose[i] for i in range(6)]

    def get_target_pose(self, x=0, y=0, z=0):
        """通过相机获取目标状态"""
        self.get_state()
        i = 0
        while i < 7:
            i += 1
            rospy.wait_for_service('/pose')
            req = poseRequest()
            req.rank = self.current_request.rank
            if x == -1:
                req.target = "zxsd"
            elif x == -2:
                req.target = "yb"
            else:
                req.target = self.current_request.target
            self.target_pose = self.client(req)
            print(f"target_pose = {self.target_pose}")
            if self.target_pose.pose[0] ** 2 + self.target_pose.pose[1] ** 2 > self.current_pose[0] ** 2 + \
                    self.current_pose[1] ** 2 and self.target_pose.pose[0] != 10.0:
                self.pose_flag = True
                print(f"target_pose effective")
                break
            # elif self.target_pose.pose[0]**2 + self.target_pose.pose[1]**2 + self.target_pose.pose[2]**2 > 0.9**2:
            #     print(f"超过工作半径0.9m")
            else:
                if x**2 + y**2 + z**2 > 0:
                    if i == 6:
                        break
                    else:
                        self.position_dxyz(x, y, z * (-1) ** (i + 1))

    def get_target_pose_1(self, x=0, y=0, z=0):
        """通过相机获取目标状态"""
        self.get_state()
        i = 0
        while i < 7:
            i += 1
            rospy.wait_for_service('/pose')
            self.target_pose_1 = self.client(self.current_request.target)
            print(f"target_pose_1 = {self.target_pose_1}")
            if self.target_pose_1.pose[0] ** 2 + self.target_pose_1.pose[1] ** 2 > self.current_pose[0] ** 2 + \
                    self.current_pose[1] ** 2 and self.target_pose_1.pose[0] != 10.0:
                self.pose_flag = True
                print(f"target_pose_1 effective")
                break
            # elif self.target_pose.pose[0]**2 + self.target_pose.pose[1]**2 + self.target_pose.pose[2]**2 > 0.9**2:
            #     print(f"超过工作半径0.9m")
            else:
                if x**2 + y**2 + z**2 > 0:
                    if i == 6:
                        break
                    else:
                        self.position_dxyz(x, y, z * (-1) ** (i + 1))

    def GET_TARGET_POSE(self):
        self.get_target_pose()
        self.get_target_pose_1()

        print(f"{self.target_pose.pose[1]}")
        print(f"{self.target_pose_1.pose[1]}")
        dy = abs(self.target_pose.pose[1] - self.target_pose_1.pose[1])
        print(f"{dy}")
        if abs(dy) > 0.001:
            self.get_target_pose()

    def get_state(self):
        """获取机械臂状态"""
        self.test_Get_Arm_Base_State_pub.publish(self.command)
        rospy.sleep(0.5)
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

    def dxy(self, x, y, mod=0):
        a = self.current_request.direction % 10
        b = self.current_request.angle

        if mod == 0:
            dx = x * np.cos(np.pi * a / 2) + y * np.sin(np.pi * a / 2)
            dy = - x * np.sin(np.pi * a / 2) + y * np.cos(np.pi * a / 2)
        if mod == 1:
            dx = x * np.cos(np.pi * a / 2 + b) + y * np.sin(np.pi * a / 2 + b)
            dy = - x * np.sin(np.pi * a / 2 + b) + y * np.cos(np.pi * a / 2 + b)
        return dx, dy

    def check_target_found(self, num = 0):
        if self.target_pose.pose[0] > 2:
            print(f"failure: target not found")
            self.position_dxyz(-0.1, 0, 0)
            self.init_position(self.current_request.direction + num)
            self.action = False
            return True
        return False
    # -------------------------------------------功能函数-------------------------------------------
    def init_position(self, num):
        """初始位置 和 相对的区域"""
        if num == 0:
            """
            夹爪朝前
            x, y, z = 0.37, 0, 0.5
            rx, ry, rz = 3.14, -1.57, 0
            """
            joint = [-0.002827, -0.801839, 2.048196, 0.010123, 0.324143, 3.122795]
            self.move_j(joint, 0.25, 1)
            self.arrival_flag = 0
            print("到达初始姿态")

        if num == 1:
            """
            夹爪朝右
            x, y, z = 0, -0.37, 0.5
            rx, ry, rz = 1.57, -1.57, 0
            """
            joint = [-1.573623, -0.801839, 2.048196, 0.010123, 0.324143, 3.122795]
            self.move_j(joint, 0.25, 1)
            self.arrival_flag = 0
            print("到达右侧初始姿态")

        if num == 2:
            """
            夹爪朝后
            x, y, z = -0.5, 0.05, 0.48
            rx, ry, rz = 0, -1.57, 0
            """
            joint = [-0.275587, 0.931552, -2.471892, 1.463511, -0.277263, -1.460073]
            self.move_j(joint, 0.25, 1)
            self.arrival_flag = 0
            print("到达后方初始姿态")

        if num == 3:
            """
            夹爪朝左
            x, y, z = 0, 0.37, 0.5
            rx, ry, rz = 0, -1.57, -1.57
            """
            joint = [90, -45.94, 117.354, 0.579, 18.569, 178.92]
            self.move_j(joint, 0.25)
            self.arrival_flag = 0
            print("到达左侧初始姿态")

        if num == 10:
            joint = [98.881, 30.647, -130.13, 91.446, -81.143, -99.588]
            self.move_j(joint, 0.25)
            self.arrival_flag = 0

        if num == 11:
            joint = [90, 64.256, -131.059, 0.16, -23.196, -0.136]
            self.move_j(joint, 0.25)
            self.arrival_flag = 0

        if num == 12:
            joint = [0, 64.256, -131.059, 0.16, -23.196, -0.136]
            self.move_j(joint, 0.25)
            self.arrival_flag = 0

        if num == 13:
            joint = [-90, 64.256, -131.059, 0.16, -23.196, -0.136]
            self.move_j(joint, 0.25)
            self.arrival_flag = 0

        if num == 20:
            joint = [90, 85, -174, -15, 100, 165]
            self.move_j(joint, 0.25)

            joint = [90, 100, -174, -15, 158, 165]
            self.move_j(joint, 0.25)
            self.arrival_flag = 0

    def position_dxyz(self, x, y, z, mod=0, v=0.25, move = 0):
        """姿态不变，位移相对距离"""

        self.get_state()
        if mod == 0:
            dx, dy = self.dxy(x, y)
            pose = [self.current_pose[0] + dx, self.current_pose[1] + dy, self.current_pose[2] + z] \
                   + self.current_pose[-4:]

        if mod == 1:
            if self.current_request.direction == 1 or self.current_request.direction == 2:
                dx, dy = self.dxy(x, y)
                pose = [self.current_pose[0] + dx, self.current_pose[1] + dy, self.current_pose[2] + z] \
                       + [self.current_pose_n[3] - self.current_request.angle, self.current_pose_n[4],
                          self.current_pose_n[5]]
        if mod == 2:
            if self.current_request.direction == 1 or self.current_request.direction == 2:
                angle = -self.current_request.angle
                xx = np.cos(angle) * x - np.sin(angle) * y
                yy = np.sin(angle) * x + np.cos(angle) * y
                dx, dy = self.dxy(xx, yy)
                pose = [self.current_pose[0] + dx, self.current_pose[1] + dy, self.current_pose[2] + z] \
                       + [self.current_pose_n[3], self.current_pose_n[4], self.current_pose_n[5]]
        if move == 0:
            self.move_jp(pose, v)
        if move == 1:
            self.move_l(pose, v)
        self.arrival_flag = 0

    def delta_joint(self, delta_joint, v = 0.25):
        self.get_state()
        joint = self.current_joint
        joint[0] += np.radians(delta_joint[0])
        joint[1] += np.radians(delta_joint[1])
        joint[2] += np.radians(delta_joint[2])
        joint[3] += np.radians(delta_joint[3])
        joint[4] += np.radians(delta_joint[4])
        joint[5] += np.radians(delta_joint[5])
        self.move_j(joint, v, 1)
        self.arrival_flag = 0

    def position_teach(self, axis='z', direction='pos', threshold=5.0, v=5, timeout=30.0):
        """
        通用位置示教函数
        axis: 轴向 'x', 'y', 'z'
        direction: 方向 'pos' 正向, 'neg' 负向
        threshold: 力阈值(N)
        timeout: 超时时间(秒)
        """
        try:
            # # 等待六维力传感器数据就绪
            # rospy.loginfo(f"等待六维力传感器数据...")
            # start_time = rospy.Time.now()
            # while not self.six_force_ready and (rospy.Time.now() - start_time).to_sec() < 5.0:
            #     rospy.sleep(0.1)

            # if not self.six_force_ready:
            #     raise Exception("六维力传感器数据未就绪")

            # 清空六维力数据
            rospy.loginfo("清空六维力数据...")

            self.clear_result_received = False
            self.clear_success = False

            self.clear_force_data_pub.publish(Empty())
            # 等待回调结果，最多等待3秒
            clear_timeout = 3.0
            start_time = rospy.Time.now()
            while not self.clear_result_received and (rospy.Time.now() - start_time).to_sec() < clear_timeout:
                rospy.sleep(0.1)  # 短暂睡眠，避免CPU占用过高

            if not self.clear_result_received:
                raise Exception("清空六维力数据超时，未收到响应")

            if not self.clear_success:
                raise Exception("清空六维力数据失败")
            else:
                print(f"清空六维力数据")
            # rospy.loginfo(f"开始沿工具系{axis.upper()}轴{direction}向示教...")

            # 开始位置示教
            pos_teach = Pos_Teach()
            pos_teach.teach_type = axis
            pos_teach.direction = direction
            pos_teach.v = v
            self.arm_position_teach_pub.publish(pos_teach)

            # 监控力传感器数据
            start_time = rospy.Time.now()
            count = 0
            triggered = False

            while (rospy.Time.now() - start_time).to_sec() < timeout:
                six_force = rospy.wait_for_message("/rm_driver/UdpSixZeroForce", Six_Force)
                fx, fy, fz = six_force.force_Fx, six_force.force_Fy, six_force.force_Fz
                mx, my, mz = six_force.force_Mx, six_force.force_My, six_force.force_Mz

                if count % 10 == 0:
                    rospy.loginfo(f"六维力数据 - Fx: {fx:.2f}N, Fy: {fy:.2f}N, Fz: {fz:.2f}N")
                    rospy.loginfo(f"六维力数据 - Mx: {mx:.2f}N, My: {my:.2f}N, Mz: {mz:.2f}N")

                # 根据轴向选择检测的力分量
                if axis == 'x' and abs(fx) >= threshold:
                    rospy.loginfo(f"检测到X轴力达到阈值: {fx:.2f}N，停止示教")
                    triggered = True
                elif axis == 'y' and abs(fy) >= threshold:
                    rospy.loginfo(f"检测到Y轴力达到阈值: {fy:.2f}N，停止示教")
                    triggered = True
                elif axis == 'z' and abs(fz) >= threshold:
                    rospy.loginfo(f"检测到Z轴力达到阈值: {fz:.2f}N，停止示教")
                    triggered = True

                if triggered:
                    stop_teach = Stop_Teach()
                    stop_teach.command = ''
                    self.arm_teach_stop_pub.publish(stop_teach)
                    break

                count += 1
                rospy.sleep(0.1)

            if not triggered:
                stop_teach = Stop_Teach()
                stop_teach.command = 'stop'
                self.arm_teach_stop_pub.publish(stop_teach)
                raise Exception("位置示教超时")

            # 获取最终的力数据
            rospy.sleep(0.5)
            rospy.loginfo(f"最终六维力数据 - Fx: {fx:.2f}N, Fy: {fy:.2f}N, Fz: {fz:.2f}N")
            rospy.loginfo(f"最终六维力数据 - Mx: {mx:.2f}N, My: {my:.2f}N, Mz: {mz:.2f}N")
            rospy.loginfo("位置示教完成")
            return True

        except Exception as e:
            rospy.logerr(f"位置示教失败: {e}")
            stop_teach = Stop_Teach()
            stop_teach.command = 'stop'
            self.arm_teach_stop_pub.publish(stop_teach)
            return False

    def set_tool_v(self, mod, block=True):
        # 0-0v 1-12v
        if mod == 0:
            a = self.robot.Set_Tool_Voltage(0, block)  # 0-0v 1-5v 2-12v 3-24v
            if a == 0:
                print("工具端电源设置为0")

        if mod == 1:
            a = self.robot.Set_Tool_Voltage(2, block)
            if a == 0:
                print("工具端电源设置为12")
               

    # -------------------------------------------按钮-------------------------------------------
    def press(self):
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0
        if self.arrival_flag == i:
            self.get_state()
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1])
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2]
            # q = euler_to_quaternion(0, -1.57, 0)
            moveJ_P_TargetPose.Pose.orientation.x = self.target_pose.pose[3]
            moveJ_P_TargetPose.Pose.orientation.y = self.target_pose.pose[4]
            moveJ_P_TargetPose.Pose.orientation.z = self.target_pose.pose[5]
            moveJ_P_TargetPose.Pose.orientation.w = self.target_pose.pose[6]
            moveJ_P_TargetPose.trajectory_connect = 0
            moveJ_P_TargetPose.speed = 0.3
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0.06, 0)
            moveJ_P_TargetPose.Pose.position.x += dx
            moveJ_P_TargetPose.Pose.position.y += dy
            moveJ_P_TargetPose.speed = 0.2
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        self.arrival_flag = 0
        print("操作完成")

    def control_button_q(self):
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0

        if self.arrival_flag == i:
            dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1])
            moveJ_P_TargetPose.Pose.position.x = self.target_pose.pose[0] + dx
            moveJ_P_TargetPose.Pose.position.y = self.target_pose.pose[1] + dy
            moveJ_P_TargetPose.Pose.position.z = self.target_pose.pose[2] + self.current_request.offset[2]
            q = euler_to_quaternion(1.4, 0, 0)
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

    # -------------------------------------------压板投入-------------------------------------------
    def control_switch_open(self, l=0.036):
        # 压板 投入

        self.get_state()
        dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1] + 0.0013, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] - l + 0.0008] \
               + self.current_pose[-4:]
        self.move_jp(pose, 0.25)

        self.set_gripper(250)
        rospy.sleep(1.5)

        self.get_state()
        joint = self.current_joint
        joint[5] += -np.pi / 2
        self.move_j(joint, 0.15, 1)

        self.get_state()
        dx, dy = self.dxy(0.063, 0, 1)
        pose = [self.current_pose[0] + dx, self.current_pose[1] + dy, self.current_pose[2]] \
               + self.current_pose[-4:]
        self.move_l(pose, 0.15)

        self.close_gripper()
        rospy.sleep(1.5)

        self.get_state()
        joint = self.current_joint
        joint[5] += np.pi / 2
        self.move_j(joint, 0.15, 1)

        self.set_gripper(130)
        rospy.sleep(1.5)

        self.get_state()
        dx, dy = self.dxy(-0.02, 0, 1)
        pose = [self.current_pose[0] + dx, self.current_pose[1] + dy, self.current_pose[2]] \
               + self.current_pose[-4:]
        self.move_l(pose, 0.15)

        self.close_gripper()
        rospy.sleep(1.5)

        self.get_state()
        joint = self.current_joint
        joint[5] += -np.pi *2 / 3
        self.move_j(joint, 0.15, 1)

        # 压板投入
        self.get_state()
        pose = [self.current_pose[0], self.current_pose[1], self.current_pose[2] - 0.022] \
               + self.current_pose[-4:]
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(0, 0.026, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(0.041, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(0, -0.002, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += 0.034
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(0, -0.018, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += 0.011
        pose[3] = 1.57
        pose[4] = 0
        pose[5] = 0
        pose = pose[:6]
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(0, 0.02, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.16, -0.05, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += -0.15
        pose[3] = 1.57
        pose[4] = -1.57
        pose[5] = 0
        self.move_l(pose, 0.25)

        self.get_state()
        joint = self.current_joint
        joint[5] += -4.5/180*np.pi
        self.move_j(joint, 0.15, 1)

        self.arrival_flag = 0
        print("操作完成")

    def control_switch_open2(self, l=0.036):
        # 压板 投入2
        self.get_state()
        dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1] - 0.001, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2]] \
               + self.current_pose[-4:]
        self.move_jp(pose, 0.15)

        self.set_gripper(250)
        rospy.sleep(1.5)

        dx, dy = self.dxy(0.063, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        self.close_gripper()
        rospy.sleep(0.5)

        self.get_state()
        joint = self.current_joint
        joint[5] += np.pi / 2
        self.move_j(joint, 0.15, 1)

        self.set_gripper(130)
        rospy.sleep(1.5)

        self.get_state()
        dx, dy = self.dxy(-0.1, 0, 1)
        pose = [self.current_pose[0] + dx, self.current_pose[1] + dy, self.current_pose[2]] \
               + self.current_pose[-4:]
        self.move_l(pose, 0.15)

        self.close_gripper()
        rospy.sleep(1.5)

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
            moveJ_P_TargetPose.speed = 0.25
            self.moveJ_P_pub.publish(moveJ_P_TargetPose)
        while self.arrival_flag == 0 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        self.set_gripper(230)
        rospy.sleep(1.5)

        if self.arrival_flag == 1:
            self.get_state()
            dx, dy = self.dxy(0.064, 0)
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
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        self.close_gripper()
        rospy.sleep(1.5)

        if self.arrival_flag == 2:
            self.get_state()
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[5] += -np.pi / 2
            moveJ_TargetPose.speed = 0.13
            self.moveJ_pub.publish(moveJ_TargetPose)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        self.set_gripper()
        rospy.sleep(1.5)

        if self.arrival_flag == 3:
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
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        self.close_gripper()
        rospy.sleep(1.5)

        if self.arrival_flag == 4:
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
        while self.arrival_flag == 4 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        if self.arrival_flag == 5:
            dx, dy = self.dxy(0.043, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 5 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        if self.arrival_flag == 6:
            dx, dy = self.dxy(0, 0.03)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 6 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        if self.arrival_flag == 7:
            dx, dy = self.dxy(-0.13, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.trajectory_connect = 0
            moveL_TargetPose.speed = 0.15
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 7 and not rospy.is_shutdown():
            pass
        rospy.sleep(0.1)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------屏柜门正面开-------------------------------------------
    def control_door1_open_right_q(self):
        # 开屏柜门 右拉 正面

        self.get_state()
        dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1], 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2]] \
               + self.current_pose[-4:]
        self.move_jp(pose, 0.2)

        dx, dy = self.dxy(0.072, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2)

        dx, dy = self.dxy(-0.052, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2)

        dx, dy = self.dxy(0, 0.04, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2)

        pose[2] += 0.04
        self.move_l(pose, 0.2)

        dx, dy = self.dxy(0.03, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2)

        dx, dy = self.dxy(0.005, -0.068, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2, 1)

        pose[2] += -0.02
        self.move_l(pose, 0.2)

        pose[2] += 0.02
        self.move_l(pose, 0.2)

        dx, dy = self.dxy(0.005, -0.018, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2, 1)

        dx, dy = self.dxy(-0.01, 0, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += 0.06
        self.move_l(pose, 0.1)

        pose[2] += -0.021
        self.move_l(pose, 0.2)

        dx, dy = self.dxy(-0.005, 0, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += 0.02
        self.move_l(pose, 0.1)

        pose[2] += -0.15
        self.move_l(pose, 0.2)

        dx, dy = self.dxy(-0.15, 0.15, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------屏柜门正面关-------------------------------------------
    def control_door1_close_right_q4(self):
        # 关右屏柜门 正面 第1步

        joint = [0, -78, 10, 90, 0, -90]
        self.move_j(joint, 0.2)

        joint = [11, -78, 10, 90, -18, -90]
        deg = np.rad2deg(self.current_request.angle)
        joint[0] += -deg
        joint[4] += deg * 1.5
        self.move_j(joint, 0.06)

        dx, dy = self.dxy(-0.68 * np.sin((120 - 15 * 1) * np.pi / 180) + 0.06,
                          0.68 * np.cos((120 - 15 * 1) * np.pi / 180) - 0.765, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.23] \
               + [-0.714 - self.current_request.angle, -1.13, 1.65]
        self.move_jp(pose, 0.05)

        dx, dy = self.dxy(-0.68 * np.sin((120 - 15 * 2) * np.pi / 180) + 0.04,
                          0.68 * np.cos((120 - 15 * 2) * np.pi / 180) - 0.7555, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.23] \
               + [-0.714 - self.current_request.angle, -1.13, 1.65]
        self.move_jp(pose, 0.05)

        dx, dy = self.dxy(-0.68 * np.sin((120 - 15 * 3) * np.pi / 180) + 0.06,
                          0.68 * np.cos((120 - 15 * 3) * np.pi / 180) - 0.7555, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.23] \
               + [1.57 - self.current_request.angle, -1.57, 0]
        self.move_jp(pose, 0.08)

        dx, dy = self.dxy(-0.68 * np.sin((120 - 15 * 4) * np.pi / 180) + 0.03,
                          0.68 * np.cos((120 - 15 * 4) * np.pi / 180) - 0.7555, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.23] \
               + [1.57 - self.current_request.angle, -1.57, 0]
        self.move_jp(pose, 0.08)

        dx, dy = self.dxy(-0.68 * np.sin((120 - 15 * 5) * np.pi / 180) + 0.05,
                          0.68 * np.cos((120 - 15 * 5) * np.pi / 180) - 0.7555, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.23] \
               + [1.57 - self.current_request.angle, -1.57, 0]
        self.move_jp(pose, 0.08)

        dx, dy = self.dxy(-0.68 * np.sin((120 - 15 * 6) * np.pi / 180),
                          0.68 * np.cos((120 - 15 * 6) * np.pi / 180) - 0.7555, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.23] \
               + [1.57 - self.current_request.angle, -1.57, 0]
        self.move_jp(pose, 0.08)

        dx, dy = self.dxy(-0.14, self.current_request.offset[1] - 0.01, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.175] \
               + [1.57 - self.current_request.angle, -1.57, 0]
        self.move_jp(pose, 0.08)

        self.get_state()
        joint = self.current_joint
        joint[5] += -np.pi / 2
        self.move_j(joint, 0.3, 1)

        self.get_target_pose(-0.01, 0.0, -0.01)
        if abs(self.target_pose.pose[0]) > 3:
            self.GET_TARGET_POSE()

        self.get_state()
        joint = self.current_joint
        joint[5] += np.pi * 3 / 4
        self.move_j(joint, 0.3, 1)

        self.get_state()
        dx, dy = self.dxy(0.011, self.current_request.offset[1] - 0.05, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.18] \
               + self.current_pose[-4:]
        self.move_jp(pose, 0.1)

        dx, dy = self.dxy(0.001, 0, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += -0.18
        self.move_l(pose, 0.2)

        dx, dy = self.dxy(-0.01, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.25)

        dx, dy = self.dxy(-0.014, -0.05, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.25)

        self.get_state()
        joint = self.current_joint
        joint[5] += -np.pi * 3 / 4
        self.move_j(joint, 0.3, 1)

        self.get_state()
        dx, dy = self.dxy(-0.15, 0.06, 1)
        pose = [pose[0] + dx, pose[1] + dy, pose[2] + 0.12] \
               + self.current_pose[-4:]
        self.move_l(pose, 0.25)

        self.arrival_flag = 0
        print("操作完成")

    def control_door1_close2_right(self):
        # 关右屏柜门 正面 第2步
        self.get_state()
        dx, dy = self.dxy(self.current_request.offset[0] - 0.09, self.current_request.offset[1] - 0.078, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.052] \
               + self.current_pose[-4:]
        self.move_jp(pose, 0.2)

        dx, dy = self.dxy(0.065, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2)

        dx, dy = self.dxy(0, 0.0605, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(0, -0.0205, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(-0.062, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(0, 0.032, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(0.056, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        pose[2] += -0.02
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(0.048, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.21, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        self.arrival_flag = 0
        print("操作完成")

    def control_door1_close2_right_b(self):

        self.get_state()
        dx, dy = self.dxy(self.current_request.offset[0] - 0.09, self.current_request.offset[1] + 0.05, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.052] \
               + self.current_pose[-4:]
        self.move_jp(pose, 0.25)

        dx, dy = self.dxy(0.065, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.25)

        dx, dy = self.dxy(0, -0.03, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.25)

        dx, dy = self.dxy(-0.065, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.25)

        self.get_state()
        dx, dy = self.dxy(self.current_request.offset[0] - 0.09, self.current_request.offset[1] - 0.078, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.052] \
               + self.current_pose[-4:]
        self.move_jp(pose, 0.25)

        dx, dy = self.dxy(0.065, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.25)

        dx, dy = self.dxy(0, 0.0605, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(0, -0.0205, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(-0.062, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(0, 0.0356, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        dx, dy = self.dxy(0.056, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        pose[2] += -0.02
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(0.049, -0.004, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.21, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.15)

        self.arrival_flag = 0
        print("操作完成")

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
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 2:
            moveL_TargetPose.Pose.position.z += -0.02
            moveL_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 2 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 3:
            dx, dy = self.dxy(0.045, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 3 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == 4:
            dx, dy = self.dxy(-0.1, 0)
            moveL_TargetPose.Pose.position.x += dx
            moveL_TargetPose.Pose.position.y += dy
            moveL_TargetPose.speed = 0.1
            self.moveL_pub.publish(moveL_TargetPose)
        while self.arrival_flag == 4 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------防火门-------------------------------------------
    def control_door3_open_n(self, l=0.4):
        # 防火门 开
        self.get_state()
        dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1], 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2]] \
               + self.current_pose[-4:]
        self.move_jp(pose, 0.2)

        self.get_state()
        joint = self.current_joint
        joint[5] += -np.pi / 2 + np.pi / 9
        self.move_j(joint, 0.3, 1)

        self.set_gripper(400)

        self.get_state()
        pose = self.current_pose
        dx, dy = self.dxy(0.083, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        self.close_gripper()
        rospy.sleep(1.0)
        
        dx, dy = self.dxy(0, 0.015, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += -0.101
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.04, 0)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        pose[2] += 0.098
        self.move_l(pose, 0.1)

        self.get_state()
        joint = self.current_joint
        joint[5] += -np.pi / 9
        self.move_j(joint, 0.3, 1)

        l = 0.41
        theta = np.radians(25)
        x = l * np.sin(theta)
        y = l * (1 - np.cos(theta))
        self.get_state()
        pose = self.current_pose
        dx, dy = self.dxy(-x + 0.03, -y, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        self.set_gripper(600)
        rospy.sleep(1.2)

        dx, dy = self.dxy(-0.15, 0)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        self.close_gripper()

        dx, dy = self.dxy(0, 0.25)
        pose[0] += dx
        pose[1] += dy
        pose[2] += 0.18
        pose[3] = 1.57
        pose[4] = 0
        pose[5] = -2
        pose = pose[:6]
        self.move_jp(pose, 0.2)

        dx, dy = self.dxy(0.24, 0)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.25)

        dx, dy = self.dxy(0, -0.18)
        pose[0] += dx
        pose[1] += dy
        pose[5] = -2.3
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.25, -0.18)
        pose[0] += dx
        pose[1] += dy
        pose[5] = -2.5
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.05, -0.25)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(0, 0.1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.25)

        dx, dy = self.dxy(0.51, 0.35)
        pose[0] += dx
        pose[1] += dy
        pose[2] += -0.04
        pose[3] = -1.527
        pose[4] = -0.114
        pose[5] = 2.149
        self.move_jp(pose, 0.25)

        dx, dy = self.dxy(0, 0.1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.1, 0)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.2, 0.1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.15, 0.2)
        pose[0] += dx
        pose[1] += dy
        pose[3] = -1.544
        pose[4] = -0.119
        pose[5] = 2.588
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.05, 0.2)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(0, -0.15)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.25)

        self.arrival_flag = 0
        print("操作完成")

    def control_door3_close(self):
        # 关右屏柜门 正面 第1步
        moveJ_TargetPose = MoveJ()
        moveJ_P_TargetPose = MoveJ_P()
        moveL_TargetPose = MoveL()
        i = 0


        if self.arrival_flag == i:
            joint = [83.932, -37.606, -59.071, 89.246, 83.978, -83.281]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.25
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
            self.robot.Movel_Cmd(next_pose, 25, 0, trajectory_connect=0, block=True)
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
            moveJ_TargetPose.joint[4] += np.pi / 5
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            moveJ_TargetPose.joint[0] += -np.pi / 5
            moveJ_TargetPose.joint[4] += -np.pi / 8
            moveJ_TargetPose.speed = 0.07
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

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
            self.robot.Movel_Cmd(next_pose, 15, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.04, 0)
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            next_pose[0] = self.target_pose.pose[0] + dx
            self.robot.Movel_Cmd(next_pose, 15, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.1, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movel_Cmd(next_pose, 30, 0, trajectory_connect=0, block=True)
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
            joint = [-87.74, -29.267, -67.636, 97.194, -86.467, -100.38]
            joint_rad = [math.radians(angle) for angle in joint]
            moveJ_TargetPose.joint = joint_rad
            moveJ_TargetPose.speed = 0.3
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
            self.robot.Movel_Cmd(next_pose, 25, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            self.get_state()
            rospy.sleep(0.5)
            moveJ_TargetPose.joint = self.current_joint
            moveJ_TargetPose.joint[4] += -np.pi / 5
            moveJ_TargetPose.speed = 0.1
            self.moveJ_pub.publish(moveJ_TargetPose)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            moveJ_TargetPose.joint[0] += np.pi / 4.5
            moveJ_TargetPose.joint[4] += np.pi / 7
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
            self.robot.Movel_Cmd(next_pose, 20, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(0.04, 0)
            current_pose = self.robot.Get_Current_Arm_State()[2]
            next_pose = copy.deepcopy(current_pose)
            next_pose[0] = self.target_pose.pose[0] +dx
            self.robot.Movej_P_Cmd(next_pose, 15, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        if self.arrival_flag == i:
            dx, dy = self.dxy(-0.1, 0)
            next_pose[0] += dx
            next_pose[1] += dy
            self.robot.Movel_Cmd(next_pose, 25, 0, trajectory_connect=0, block=True)
            i += 1
        while self.arrival_flag == i - 1 and not rospy.is_shutdown():
            pass
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------左端子箱开-------------------------------------------
    def control_door4_open_1(self):
        # 开端子门 右拉 正面
        self.get_state()
        dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1], 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2]] \
               + self.current_pose[-4:]
        self.move_jp(pose, 0.2)

        dx, dy = self.dxy(0.055, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2)

        dx, dy = self.dxy(0, -0.07, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(0, -0.02, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += 0.055
        self.move_l(pose, 0.1)

        pose[2] += -0.03
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.08, 0, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += 0.045
        self.move_l(pose, 0.1)

        self.set_gripper(400)
        
        self.get_state()
        joint = self.current_joint
        joint[5] += -np.pi / 2
        self.move_j(joint, 0.3, 1)

        self.get_state()
        pose = self.current_pose
        dx, dy = self.dxy(0.075, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        self.set_gripper(180)
        rospy.sleep(1.0)

        dx, dy = self.dxy(-0.02, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        self.set_gripper(130)
        rospy.sleep(1.0)

        l = 0.41
        theta = np.radians(25)
        x = l * np.sin(theta)
        y = l * (1 - np.cos(theta))
        dx, dy = self.dxy(-x+0.004, -y+0.01, 1)
        pose[0] += dx
        pose[1] += dy
        pose[4] += np.radians(10)
        pose[5] += np.radians(10)
        self.move_l(pose, 0.1)

        theta = np.radians(45)
        x1 = l * np.sin(theta)
        y1 = l * (1 - np.cos(theta))
        dx, dy = self.dxy(-(x1-x)+0.004, -(y1-y)-0.01, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        self.set_gripper(800)
        rospy.sleep(1.0)

        dx, dy = self.dxy(-0.08, -0.05, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.1)

        self.set_gripper(0)
        
        self.get_state()
        joint = self.current_joint
        joint[1] = 0
        joint[2] = 0
        joint[4] = 0
        self.move_j(joint, 0.15, 1)
        rospy.sleep(1.0)

        self.arrival_flag = 0
        print("操作完成")

    # -------------------------------------------左端子箱关-------------------------------------------
    def control_door4_close(self):
        # 开屏柜门 右拉 正面
        joint = [-1.573623, -0.801839, 2.048196, 0.010123, 0.324143, 3.122795]
        joint[3] += np.pi / 12
        joint[4] += -np.pi / 2
        joint[5] += np.pi / 6
        self.move_j(joint, 0.3, 1)

        joint[0] += -np.pi / 2.8
        self.move_j(joint, 0.3, 1)

        joint[4] += np.pi / 2
        self.move_j(joint, 0.3, 1)

        self.get_state()
        pose = self.current_pose
        pose[0] += -0.25 * np.sin(np.pi/3)
        pose[1] += -0.25 * np.cos(np.pi/3)
        pose[2] += -0.2
        self.move_jp(pose, 0.3)

        self.get_state()
        joint = self.current_joint
        joint[0] += np.pi / 3.2
        self.move_j(joint, 0.08, 1)

        self.get_state()
        dx, dy = self.dxy(-0.05, -0.15, 1)
        pose = self.current_pose
        pose[0] += dx
        pose[1] += dy
        pose[3] = 1.57
        pose[4] = -1.57
        pose[5] = 0
        pose = pose[:6]
        self.move_jp(pose, 0.1)

        self.get_state()
        dx, dy = self.dxy(self.current_request.offset[0] - 0.06, self.current_request.offset[1] - 0.11, 1)
        pose = [self.target_pose.pose[0] + dx, self.target_pose.pose[1] + dy,
                self.target_pose.pose[2] + self.current_request.offset[2] + 0.09] \
               + self.current_pose[-4:]
        self.move_jp(pose, 0.1)

        dx, dy = self.dxy(0.078, 0, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2)

        pose[2] += -0.07
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(0.002, 0.05, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += -0.02
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(0, 0.01, 1)
        pose[0] += dx
        pose[1] += dy
        pose[2] += -0.015
        self.move_l(pose, 0.1)

        dx, dy = self.dxy(-0.08, -0.08, 1)
        pose[0] += dx
        pose[1] += dy
        self.move_l(pose, 0.2)

        self.arrival_flag = 0

    # -------------------------------------------换装-------------------------------------------
    def change_tool_right_on(self):
        # 右 换上工装
        joint = [-26.868, 16.697, 65.412, -0.081, 97.841, 153.120]
        self.move_j(joint, 0.3)

        self.get_state()
        pose = self.current_pose
        pose[2] += -0.1
        self.move_l(pose, 0.25)

        pose[2] += -0.04
        self.move_l(pose, 0.2)

        pose[1] += -0.13
        self.move_l(pose, 0.08)

        pose[2] += 0.14
        self.move_l(pose, 0.25)

        self.get_state()
        joint = self.current_joint
        joint[4] += -np.pi / 2
        self.move_j(joint, 0.3, 1)

        self.arrival_flag = 0

    def change_tool_right_off(self):
        # 右 退下工装
        joint = [-40.115, 28.549, 49.346, -0.072, 12.039, 139.869]
        self.move_j(joint, 0.3)

        self.get_state()
        joint = self.current_joint
        joint[4] += np.pi / 2
        self.move_j(joint, 0.3, 1)

        self.get_state()
        pose = self.current_pose
        pose[2] += -0.14
        self.move_l(pose, 0.25)

        pose[1] += 0.13
        self.move_l(pose, 0.08)

        pose[2] += 0.04
        self.move_l(pose, 0.15)

        pose[2] += 0.1
        self.move_l(pose, 0.25)

        self.arrival_flag = 0

    def change_tool_left_on(self):
        # 左 换上工装
        joint = [27.851, 15.971, 66.787, -0.988, 95.166, 27.66]
        self.move_j(joint, 0.3)

        self.get_state()
        pose = self.current_pose
        pose[2] += -0.1
        self.move_l(pose, 0.25)

        pose[2] += -0.04
        self.move_l(pose, 0.15)

        pose[1] += 0.13
        self.move_l(pose, 0.08)

        pose[2] += 0.14
        self.move_l(pose, 0.25)

        self.get_state()
        joint = self.current_joint
        joint[4] += -np.pi / 2
        self.move_j(joint, 0.3, 1)

        self.arrival_flag = 0

    def change_tool_left_off(self):
        # 左 退下工装
        joint = [41.022, 28.039, 50.522, -1.418, 9.685, 40.726]
        self.move_j(joint, 0.3)

        self.get_state()
        joint = self.current_joint
        joint[4] += np.pi / 2
        self.move_j(joint, 0.3, 1)

        self.get_state()
        pose = self.current_pose
        pose[2] += -0.139
        self.move_l(pose, 0.25)

        pose[1] += -0.13
        self.move_l(pose, 0.08)

        pose[2] += 0.04
        self.move_l(pose, 0.15)

        pose[2] += 0.1
        self.move_l(pose, 0.25)

        self.arrival_flag = 0


if __name__ == '__main__':
    try:
        arm = arm_control()
        arm.run()
        # 保持节点运行
        rospy.spin()
    except rospy.ROSInterruptException:
        pass