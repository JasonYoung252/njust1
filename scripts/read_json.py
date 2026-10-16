#!/usr/bin/env python3
import os
import json
import rospy
from njust.srv import task, taskRequest, taskResponse, pose, poseResponse

def read_json_message(txt_file_path, folder_path):
    """
    从txt文件中读取json文件名，然后在指定文件夹中找到并读取json文件的消息
    
    参数:
        txt_file_path (str): 包含json文件名的txt文件路径
        folder_path (str): 包含json文件的文件夹路径
    
    返回:
        dict: 从json文件中读取的消息内容
    """
    try:
        # 1. 从txt文件中读取json文件名
        with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
            json_filename = txt_file.read().strip()  # 去除可能的空白字符

        if not json_filename:
            return json_filename, 0
        
        # 2. 构建完整的json文件路径
        json_path = os.path.join(folder_path, json_filename)
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"在文件夹 {folder_path} 中未找到文件 {json_filename}")
        
        # 3. 读取json文件内容
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            
        return json_filename, data
    
    except Exception as e:
        print(f"发生错误: {e}")
        return None

def json_to_task(data):
    """
    将JSON数据映射到task服务请求格式
    """
    req = taskRequest()
    # 标签名
    req.target = data["command_id"]
    # 操作类型（没写完）
    if data["object_type"] == "按钮":
        req.type = "button"

    # 方向 转uint8
    req.direction = int(round(data["robot_orientation_deg"] / 90)) % 256
    # 区域（没写）
    req.region = 0
    # 偏移量
    req.offset = [                         
        data["offset_from_label"]["x"],
        data["offset_from_label"]["y"],
        data["offset_from_label"]["z"]
    ]
    # 目标当前状态和目标状态
    req.now_state = 0 if data["current_state"] == "弹起" else 1
    req.next_state = 1 if data["target_state"] == "按下" else 0
    
    return req

if __name__ == "__main__":
    # 替换为你的实际路径
    txt_file_path = "/home/robot/Project-CommandReceiver/ACTIVE_COMMAND.txt"  # 包含json文件名的txt文件
    folder_path = "/home/robot/Project-CommandReceiver/received_commands"     # 包含json文件的文件夹
    
    # 初始化ROS节点
    rospy.init_node('read_json')
    rospy.sleep(1.0)

    last_name = ""
    client = rospy.ServiceProxy('/task', task)

    print(f"读指令准备完成")

    try:
        while not rospy.is_shutdown():
            json_filename, data = read_json_message(txt_file_path, folder_path)
            if json_filename == "":
                pass
            else:
                if data is not None:
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    req = json_to_task(data)
                    rospy.wait_for_service('/task')
                    target_pose = client(req)
                    with open(txt_file_path, 'w') as file:
                        file.truncate(0)   # 清空文件内容
            rospy.sleep(0.1)
    except rospy.ROSInterruptException:
        pass

    