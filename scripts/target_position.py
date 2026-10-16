#!/usr/bin/env python3
import cv2
import rospy
import threading
import numpy as np
import D435_rgb_depth as rs
import convert as cam_to_base
from paddleocr import PaddleOCR
from njust.srv import task, taskResponse, pose, poseResponse
from rm_msgs.msg import Arm_Current_State, ArmState, GetArmState_Command

pub_position = False
current_request = None
Armstate = [0, 0, 0, 0, 0, 0]
Arm_State = [0, 0, 0, 0, 0, 0, 0]
target_pose = [0, 0, 0, 0, 0, 0, 0]

command = GetArmState_Command()
command.command = "get_current_arm_state"
test_Get_Arm_Base_State_pub = rospy.Publisher("/rm_driver/GetArmState_Cmd", GetArmState_Command, queue_size=10)

# ---------- 打印OCR结果并显示 ----------
def ocr_print_and_show(image, results, target_label):
    """
    图像  OCR结果  标签
    """
    # while isinstance(results, list) and len(results) == 1:     # isinstance Python内置函数调用，判断results是否是list
    #     results = results[0]

    # 只绘制目标标签的文字框
    for i, item in enumerate(results, start=1):  # 遍历所有OCR识别结果
        box, (text, conf) = item  # box 文字框的坐标列表 (text, conf) 识别出的文本和置信度
        box_str = ', '.join(f"({int(x)},{int(y)})" for x, y in box)  # 生成字符串
        print(f"第{i}条识别结果:")
        print(f"  文字框坐标: [{box_str}]")
        print(f"  识别内容: {text}")
        print(f"  置信度: {conf:.4f}\n")
        
    #     # 将文本框转换为矩形框并绘制
    #     pts = np.array(box, dtype=np.int32)
    #     pts = pts.reshape((-1, 1, 2))
    #     # 绘制轮廓（多边形框）
    #     cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    #     # 在文字框上添加文字
    #     cv2.putText(image, text, (int(box[0][0]), int(box[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # # 显示图像，并立即返回，不阻塞程序
    # cv2.imshow("OCR Results", image)
    # cv2.waitKey(2000)  # 等待1000毫秒，立即返回，不会阻塞程序
    # cv2.destroyAllWindows()  # 关闭窗口

# -------- 显示并过滤 OCR 结果 ---------
def show_ocr_results(image, target_label, conf_threshold=0.8):
    """
    图像  标签  置信度阈值
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image  # 读取图像
    if img is None:
        raise ValueError("无法读取输入图像")
    
    # 执行 OCR
    results = ocr.ocr(img, cls=True)
    if not results:
        print("未检测到文字区域")
        return []
    
    filtered = []
    for line in results:
        if not line:
            continue

        items = line if isinstance(line[0][0], list) else [line]
        for item in items:
            _, (_, conf) = item
            if conf >= conf_threshold:
                filtered.append(item)  # 结果添加到filtered

    if not filtered:
        print("未检测到符合置信度的文字结果")
        return []
    
    # 只绘制与目标标签匹配的框
    ocr_print_and_show(img, filtered, target_label)  # 修改这里，传入目标标签
    return filtered

# ---------- 提取目标标签像素位置与类型 ----------
def get_target_pose(image, ocr_target, conf_threshold=0.6):
    image_copy = image.copy()  # # 避免修改原始图像
    filtered = show_ocr_results(image, ocr_target, conf_threshold)
    for box, (text, _) in filtered:
        text_clear = text.strip()  # text.strip()：去除文本前后空格，避免匹配错误
        if text_clear == ocr_target:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x_center = int(sum(xs)/len(xs))
            y_center = int(sum(ys)/len(ys))
            # 在图像上绘制中心点（红色圆点）
            cv2.circle(image_copy, (x_center, y_center), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.imshow("u_v_center", image_copy)
            cv2.waitKey(1000)  # 等待1000毫秒，立即返回，不会阻塞程序
            cv2.destroyAllWindows()  # 关闭窗口  
            return (x_center, y_center)
        elif 'I' in text_clear:  #防止I和1混淆
            text_replaced = text_clear.replace('I','1')
            if text_replaced == ocr_target:
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x_center = int(sum(xs)/len(xs))
                y_center = int(sum(ys)/len(ys))
                # 在图像上绘制中心点（红色圆点）
                cv2.circle(image_copy, (x_center, y_center), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.imshow("u_v_center", image_copy)
                cv2.waitKey(1000)  # 等待1000毫秒，立即返回，不会阻塞程序
                cv2.destroyAllWindows()  # 关闭窗口  
                return (x_center, y_center)
        else:  #防止l和1混淆
            text_replaced = text_clear.replace('l','1')
            if text_replaced == ocr_target:
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x_center = int(sum(xs)/len(xs))
                y_center = int(sum(ys)/len(ys))
                # 在图像上绘制中心点（红色圆点）
                cv2.circle(image_copy, (x_center, y_center), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.imshow("u_v_center", image_copy)
                cv2.waitKey(1000)  # 等待1000毫秒，立即返回，不会阻塞程序
                cv2.destroyAllWindows()  # 关闭窗口  
                return (x_center, y_center)
    print(f"未找到标签 '{ocr_target}'")
    return (None, None)

# ---------- OCR检测并转换坐标 ----------
def ocr_detection(ocr_target, camera_index=0, timeout=5):
    _, _, color_image, _, _ = realsense_processor.get_aligned_images()

    image_copy = color_image.copy()
    cv2.imshow("cs", image_copy)
    # cv2.imwrite("/home/njust/RobotArm_ws/p/2.jpg",image_copy)
    cv2.waitKey(2000)  # 等待2000毫秒，立即返回，不会阻塞程序
    cv2.destroyAllWindows()  # 关闭窗口  

    (u, v) = get_target_pose(color_image, ocr_target)
    if u is None:
        return None  # 返回None，避免解包错误
    depth_pixel = [u, v]
    _, _, _, X_cam, Y_cam, Z_cam = realsense_processor.process_frame(depth_pixel)
    print(f"{X_cam},{Y_cam},{Z_cam}")
    now_pose = get_current_arm_state()
    X_base, Y_base, Z_base = cam_to_base.convert(X_cam, Y_cam, Z_cam, *now_pose)
    global Arm_State
    label_pose = [X_base, Y_base, Z_base, Arm_State[3], Arm_State[4], Arm_State[5], Arm_State[6]]
    return label_pose

# ---------- 机械臂状态回调函数 ----------
def GetArmState_Callback(msg):
    global Armstate
    Armstate[0] = msg.Pose[0]
    Armstate[1] = msg.Pose[1]
    Armstate[2] = msg.Pose[2]
    Armstate[3] = msg.Pose[3]
    Armstate[4] = msg.Pose[4]
    Armstate[5] = msg.Pose[5]

def Get_Arm_State_Callback(msg):
    global Arm_State
    Arm_State[0] = msg.Pose.position.x
    Arm_State[1] = msg.Pose.position.y
    Arm_State[2] = msg.Pose.position.z
    Arm_State[3] = msg.Pose.orientation.x
    Arm_State[4] = msg.Pose.orientation.y
    Arm_State[5] = msg.Pose.orientation.z
    Arm_State[6] = msg.Pose.orientation.w

# ---------- 获取当前机械臂状态 ----------
def get_current_arm_state():
    global Armstate
    global Arm_State
    now_pose = Armstate
    now_pose[0] = Arm_State[0]
    now_pose[1] = Arm_State[1]
    now_pose[2] = Arm_State[2]
    print(f"arm pose is: [{now_pose[0]}, {now_pose[1]}, {now_pose[2]}, {now_pose[3]}, {now_pose[4]}, {now_pose[5]}]")
    return now_pose

# ---------- 多线程 计算位置信息 ----------
def position_thread():
    global current_request, pub_position, Arm_State, target_pose
    while True:
        if pub_position:
            target_pose = ocr_detection(current_request.target)
            pub_position = False

# ---------- 服务回调函数 ----------
def poseCallback(req):
    global current_request, pub_position, command, target_pose
    if pub_position == False:
        current_request = req
        print(f"收到目标标签: {req.target}")
        test_Get_Arm_Base_State_pub.publish(command)
        pub_position = True
        while pub_position:
            pass
        if target_pose is not None:
            print(f"target_pose:{target_pose}")
            return poseResponse(target_pose)
        else:
            return poseResponse([0, 0, 0, 0, 0, 0, 0])
    else:
        return poseResponse("wait")

def main():
    rospy.init_node('target_position')
    rospy.sleep(1.0)

    # 订阅机械臂当前状态指令(弧度+四元数)
    Arm_Base_State_sub = rospy.Subscriber("/rm_driver/ArmCurrentState", ArmState, Get_Arm_State_Callback)

    # 订阅机械臂当前状态指令(角度+欧拉角)
    Arm_New_State_sub = rospy.Subscriber("/rm_driver/Arm_Current_State", Arm_Current_State, GetArmState_Callback)

    service = rospy.Service('/pose', pose, poseCallback)
    
    # thread.start_new_thread(position_thread,())
    threading.Thread(target=position_thread, daemon=True).start()
    print("OCR Realsense 初始化完成")
    rospy.spin()

if __name__ == "__main__":
    # 初始化 OCR 实例
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en')
    # print("OCR 初始化完成")
    # 初始化 RealSense 
    realsense_processor = rs.RealSenseProcessor()
    # print("RealSense 初始化完成")

    main()
