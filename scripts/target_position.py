#!/usr/bin/env python3
import os
import csv
import json
import math
import time
import cv2
import rospy
import threading
import numpy as np
import D435_rgb_depth as rs
import convert as cam_to_base
import pose_utils
from depth_localizer import DepthLocalizer, LocalizerConfig, box_from_polygon, facing_yaw_from_normal
from paddleocr import PaddleOCR
from njust.srv import task, taskResponse, pose, poseResponse
from rm_msgs.msg import Arm_Current_State, ArmState, GetArmState_Command

pub_position = False
current_request = None
Armstate = [0, 0, 0, 0, 0, 0]
Arm_State = [0, 0, 0, 0, 0, 0, 0]
target_pose = [0, 0, 0, 0, 0, 0, 0]
detection_done = threading.Event()
service_lock = threading.Lock()

# 运行参数（main() 中从 ROS 参数服务器读取）
params = {
    "n_frames": 5,            # 每次定位采集的深度帧数
    "settle_frames": 3,       # 采集前丢弃的帧数（清掉机械臂运动前的旧帧）
    "debug_view": True,       # 是否弹出 OCR 调试窗口（原程序行为）
    "log_csv": "",            # 定位日志 CSV 路径，留空不记录
    "save_dir": "",           # 保存彩色图与原始深度帧的目录，留空不保存（用于离线复现和建数据集）
    "depth_width": 848,       # 深度分辨率（原程序 1280×720）
    "depth_height": 480,
    "visual_preset": "",      # 留空不修改；可选 high_accuracy / high_density / default
    "laser_power": "",        # 留空不修改；"max" 或数值（mW）
}

ocr = None
realsense_processor = None
localizer = None

command = GetArmState_Command()
command.command = "get_current_arm_state"
test_Get_Arm_Base_State_pub = rospy.Publisher("/rm_driver/GetArmState_Cmd", GetArmState_Command, queue_size=10)


def show_image(name, image, wait_ms):
    if params["debug_view"]:
        cv2.imshow(name, image)
        cv2.waitKey(wait_ms)
        cv2.destroyAllWindows()


# ---------- 打印OCR结果并显示 ----------
def ocr_print_and_show(image, results, target_label):
    """
    图像  OCR结果  标签
    """
    for i, item in enumerate(results, start=1):  # 遍历所有OCR识别结果
        box, (text, conf) = item  # box 文字框的坐标列表 (text, conf) 识别出的文本和置信度
        box_str = ', '.join(f"({int(x)},{int(y)})" for x, y in box)  # 生成字符串
        print(f"第{i}条识别结果:")
        print(f"  文字框坐标: [{box_str}]")
        print(f"  识别内容: {text}")
        print(f"  置信度: {conf:.4f}\n")


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

    ocr_print_and_show(img, filtered, target_label)
    return filtered


def text_matches(text, target):
    """标签匹配，容忍 OCR 常见的 I/1、l/1 混淆（与原程序规则一致）。"""
    t = text.strip()
    return t == target or t.replace('I', '1') == target or t.replace('l', '1') == target


# ---------- 提取目标标签文字框 ----------
def find_target_box(image, ocr_target, conf_threshold=0.6):
    """返回 ((u, v), polygon)；没找到时返回 (None, None)。"""
    filtered = show_ocr_results(image, ocr_target, conf_threshold)
    for box, (text, _) in filtered:
        if text_matches(text, ocr_target):
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x_center = int(sum(xs) / len(xs))
            y_center = int(sum(ys) / len(ys))
            if params["debug_view"]:
                image_copy = image.copy()
                cv2.polylines(image_copy, [np.array(box, dtype=np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                cv2.circle(image_copy, (x_center, y_center), radius=3, color=(0, 0, 255), thickness=-1)
                show_image("u_v_center", image_copy, 1000)
            return (x_center, y_center), box
    print(f"未找到标签 '{ocr_target}'")
    return None, None


# ---------- 提取目标标签像素位置（保留原接口） ----------
def get_target_pose(image, ocr_target, conf_threshold=0.6):
    center, _ = find_target_box(image, ocr_target, conf_threshold)
    return center if center is not None else (None, None)


# ---------- 记录与保存 ----------
def log_localization(target, result, cam_pose, base_xyz, facing_yaw):
    if not params["log_csv"]:
        return
    path = params["log_csv"]
    new_file = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["time", "target", "status", "method", "x_cam", "y_cam", "z_cam", "x_base", "y_base",
                            "z_base", "facing_yaw_deg", "sigma_normal_mm", "sigma_lateral_mm", "roi_valid_ratio",
                            "plane_inlier_ratio", "roi_depth_m", "message", "arm_pose"])
            p = result.point_cam if result.point_cam is not None else [float("nan")] * 3
            b = base_xyz if base_xyz is not None else [float("nan")] * 3
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), target, result.status, result.method,
                        p[0], p[1], p[2], b[0], b[1], b[2],
                        math.degrees(facing_yaw) if math.isfinite(facing_yaw) else float("nan"),
                        result.sigma_normal_m * 1000, result.sigma_lateral_m * 1000, result.roi_valid_ratio,
                        result.plane_inlier_ratio, result.roi_depth_m, result.message, json.dumps(list(cam_pose))])
    except Exception as e:
        rospy.logwarn(f"写定位日志失败：{e}")


def save_capture(target, bundle, center, polygon, result, arm_pose):
    if not params["save_dir"]:
        return
    try:
        os.makedirs(params["save_dir"], exist_ok=True)
        now = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + "_%03d" % int((now % 1) * 1000)
        stem = os.path.join(params["save_dir"], f"{stamp}_{target}")
        cv2.imwrite(stem + "_color.png", bundle.color_image)
        depth_mm = np.round(bundle.depth_stack_m / 0.001).astype(np.uint16)
        np.savez_compressed(stem + "_depth.npz", depth_mm=depth_mm)
        meta = {"target": target, "center": list(center),
                "polygon": [[float(a), float(b)] for a, b in polygon] if polygon is not None else None,
                "intrinsics": [bundle.intrinsics.fx, bundle.intrinsics.fy, bundle.intrinsics.cx, bundle.intrinsics.cy],
                "arm_pose": [float(a) for a in arm_pose],
                "status": result.status, "message": result.message}
        with open(stem + "_meta.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        rospy.logwarn(f"保存采集数据失败：{e}")


# ---------- OCR检测并转换坐标 ----------
def ocr_detection(ocr_target, camera_index=0, timeout=5):
    """
    同一次采集完成 OCR 和取深度（原程序取深度时会重新抓一帧）。
    返回 13 维 pose 列表（见 pose_utils）；失败返回 None。
    """
    bundle = realsense_processor.capture_bundle(n_frames=params["n_frames"], settle_frames=params["settle_frames"])
    color_image = bundle.color_image
    show_image("cs", color_image.copy(), 2000)

    center, polygon = find_target_box(color_image, ocr_target)
    if center is None:
        return None
    box = box_from_polygon(polygon)
    result = localizer.localize(bundle.depth_stack_m, center, box)
    now_pose = get_current_arm_state()
    save_capture(ocr_target, bundle, center, polygon, result, now_pose)

    if not result.ok:
        rospy.logwarn(f"[定位失败] {ocr_target}: {result.message}")
        log_localization(ocr_target, result, now_pose, None, float("nan"))
        return None

    X_cam, Y_cam, Z_cam = [float(a) for a in result.point_cam]
    X_base, Y_base, Z_base = cam_to_base.convert(X_cam, Y_cam, Z_cam, *now_pose)
    facing_yaw = float("nan")
    if result.normal_cam is not None:
        normal_base = cam_to_base.rotate_to_base(result.normal_cam, *now_pose)
        facing_yaw = facing_yaw_from_normal(normal_base)

    yaw_deg = math.degrees(facing_yaw) if math.isfinite(facing_yaw) else float("nan")
    rospy.loginfo(f"[定位] {ocr_target}: {result.method}, cam=({X_cam:.4f},{Y_cam:.4f},{Z_cam:.4f}) m, "
                  f"base=({X_base:.4f},{Y_base:.4f},{Z_base:.4f}) m, "
                  f"σn={result.sigma_normal_m * 1000:.2f} mm, σlat={result.sigma_lateral_m * 1000:.2f} mm, "
                  f"ROI有效率={result.roi_valid_ratio:.0%}, 平面内点={result.plane_inlier_ratio:.0%}, "
                  f"朝向={yaw_deg:.1f}°；{result.message}")
    log_localization(ocr_target, result, now_pose, (X_base, Y_base, Z_base), facing_yaw)

    global Arm_State
    quat = [Arm_State[3], Arm_State[4], Arm_State[5], Arm_State[6]]
    return pose_utils.pack_pose((X_base, Y_base, Z_base), quat, facing_yaw, result.sigma_normal_m,
                                result.sigma_lateral_m, result.roi_valid_ratio, result.plane_inlier_ratio,
                                result.status)


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
    """位置取自 ArmCurrentState（m），欧拉角取自 Arm_Current_State（rad）。返回副本，不修改全局变量。"""
    now_pose = list(Armstate)
    now_pose[0] = Arm_State[0]
    now_pose[1] = Arm_State[1]
    now_pose[2] = Arm_State[2]
    print(f"arm pose is: [{now_pose[0]}, {now_pose[1]}, {now_pose[2]}, {now_pose[3]}, {now_pose[4]}, {now_pose[5]}]")
    return now_pose


# ---------- 多线程 计算位置信息 ----------
def position_thread():
    """OCR 与相机都在这个线程里使用（与原程序一致）。"""
    global current_request, pub_position, target_pose
    while not rospy.is_shutdown():
        if pub_position:
            try:
                target_pose = ocr_detection(current_request.target)
            except Exception as e:
                rospy.logerr(f"定位过程出错：{e}")
                target_pose = None
            pub_position = False
            detection_done.set()
        else:
            time.sleep(0.01)


# ---------- 服务回调函数 ----------
def poseCallback(req):
    global current_request, pub_position, command, target_pose
    if not service_lock.acquire(blocking=False):
        # 原程序此处返回字符串 "wait"，与 float32[] 类型不符，会导致服务序列化出错
        rospy.logwarn("上一次定位尚未完成，返回无效结果")
        return poseResponse(pose_utils.failed_pose())
    try:
        current_request = req
        print(f"收到目标标签: {req.target}")
        test_Get_Arm_Base_State_pub.publish(command)
        detection_done.clear()
        pub_position = True
        detection_done.wait()
        if target_pose is not None:
            print(f"target_pose:{target_pose}")
            return poseResponse(target_pose)
        return poseResponse(pose_utils.failed_pose())
    finally:
        service_lock.release()


def load_params():
    for key in list(params.keys()):
        params[key] = rospy.get_param("~" + key, params[key])
    cfg = LocalizerConfig()
    for key in cfg.__dataclass_fields__:
        name = "~localizer/" + key
        if rospy.has_param(name):
            setattr(cfg, key, rospy.get_param(name))
    return cfg


def main():
    global localizer, realsense_processor
    rospy.init_node('target_position')
    rospy.sleep(1.0)
    cfg = load_params()
    # 初始化 RealSense（放在读取参数之后，以便通过 launch 调整相机设置）
    laser = params["laser_power"]
    if laser not in ("", "max"):
        laser = float(laser)
    realsense_processor = rs.RealSenseProcessor(depth_size=(int(params["depth_width"]), int(params["depth_height"])),
                                                visual_preset=params["visual_preset"] or None,
                                                laser_power=laser or None)
    localizer = DepthLocalizer(realsense_processor.intrinsics, cfg)

    # 订阅机械臂当前状态指令(弧度+四元数)
    Arm_Base_State_sub = rospy.Subscriber("/rm_driver/ArmCurrentState", ArmState, Get_Arm_State_Callback)

    # 订阅机械臂当前状态指令(位置+欧拉角)
    Arm_New_State_sub = rospy.Subscriber("/rm_driver/Arm_Current_State", Arm_Current_State, GetArmState_Callback)

    service = rospy.Service('/pose', pose, poseCallback)

    threading.Thread(target=position_thread, daemon=True).start()
    print("OCR Realsense 初始化完成")
    rospy.spin()


if __name__ == "__main__":
    # 初始化 OCR 实例
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en')
    main()
