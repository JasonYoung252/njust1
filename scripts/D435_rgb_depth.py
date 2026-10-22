import pyrealsense2 as rs
import numpy as np
import cv2
import math

class RealSenseProcessor:
    def __init__(self, fps=30):
        """
        初始化 RealSense 管道和流设置
        :param width: 图像宽度
        :param height: 图像高度
        :param fps: 帧率
        """
        # 初始化管道
        self.pipeline = rs.pipeline()

        # 配置流
        self.config = rs.config()
        # 配置 RGB流     最高1920×1080
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, fps)  # RGB流
        # 配置 深度流     最高1280×720
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)   # 深度流

        # 启动管道
        self.pipeline.start(self.config)

        # 获取get_3d_camera_coordinate机的内参（焦距和主点）
        profile = self.pipeline.get_active_profile()  # 获取相机的活跃配置
        depth_stream = profile.get_stream(rs.stream.depth)  # 获取深度流的配置
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()  # 提取内参

        # 获取相机的内参：焦距和主点
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx  # 主点（cx）
        self.cy = intrinsics.ppy  # 主点（cy）

        # 深度图原始值 * depth_scale = 米（D435 通常为 0.001）
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        # 对齐器只需创建一次
        self.align = rs.align(rs.stream.color)

    def calculate_average_depth_in_circle(self, depth_image, center_x, center_y, radius=10):
        """
        计算以圆心为中心，半径为 radius 的圆形区域内的平均深度
        :param depth_image: 深度图像
        :param center_x: 圆心的 x 坐标
        :param center_y: 圆心的 y 坐标
        :param radius: 圆的半径
        :return: 圆形区域的平均深度
        """
        center_x = int(center_x)
        center_y = int(center_y)
        height, width = depth_image.shape
        total_depth = 0
        valid_pixels = 0

        # 遍历圆形区域内的像素
        for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                # 计算当前像素到圆心的距离
                distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                # 如果当前像素在圆形区域内且深度值不为0
                if distance <= radius and depth_image[y, x] != 0:
                    total_depth += depth_image[y, x]
                    valid_pixels += 1

        # 计算平均深度，如果没有有效像素，则返回0
        if valid_pixels > 0:
            return total_depth / valid_pixels
        else:
            return 0
    
    def get_xyz_to_uv(self, x,y,z):
        """
        计算空间坐标到相机坐标的转换矩阵
        :param x: 空间坐标 x
        :param y: 空间坐标 y
        :param z: 空间坐标 z
        :return: 相机坐标系到空间坐标系的转换矩阵
        """
        frames = self.pipeline.wait_for_frames()  #从 RealSense 摄像头获取一帧数据
        depth_frame = frames.get_depth_frame()  # 提取深度帧
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics  #从深度帧中获取相机的内参，ppx、ppy
        u = (depth_intrinsics.fx * x / z) + depth_intrinsics.ppx  # 计算像素坐标 u
        v = (depth_intrinsics.fy * y / z) + depth_intrinsics.ppy  # 计算像素坐标 v
        pixel = [int(u), int(v)]
        return u, v, pixel
    
    def get_aligned_images(self, verbose=True):
        """
        从Intel RealSense摄像头获取对齐后的RGB图像和深度图像
        """
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
    
        # 相机参数的获取
        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        
        '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                            'ppx': intr.ppx, 'ppy': intr.ppy,
                            'height': intr.height, 'width': intr.width,
                            'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                            }'''
        # 保存内参到本地
        # with open('./intrinsics.json', 'w') as fp:
        # json.dump(camera_parameters, fp)
        #######################################################
    
        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
        depth_image_3d = np.dstack(
            (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
        color_image = np.asanyarray(color_frame.get_data())  # RGB图
        if verbose:
            print("获取图片")
        # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame

    def get_3d_camera_coordinate(self,depth_pixel, aligned_depth_frame, depth_intrin):
        """
        将图像中的某个像素点转换到相机坐标系下的三维空间坐标
        depth_pixel           像素点坐标
        aligned_depth_frame   齐帧中的depth帧
        depth_intrin          深度参数
        dis                   深度
        camera_coordinate     坐标
        """
        x = depth_pixel[0]
        y = depth_pixel[1]
        print('depth_pixel: ',depth_pixel)
        dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
        print ('depth: ',dis)       # 深度单位是m
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
        # print ('camera_coordinate: ',camera_coordinate)
        print('camera_coordinate: ',camera_coordinate)
        return dis, camera_coordinate[0], camera_coordinate[1], camera_coordinate[2]

    def measure_pixel(self, depth_pixel, aligned_depth_frame, depth_intrin, depth_image, radius=2):
        """
        在【同一帧】对齐深度上，用 (2r+1)x(2r+1) 窗口内有效深度的中位数反投影到相机坐标系。
        相比单像素 get_distance：剔除了 0 值空洞，对边缘和噪点更稳健。
        :return: (深度 m, 有效像素数, [X, Y, Z] m)；窗口内无有效深度时返回 None
        """
        u = int(round(depth_pixel[0]))
        v = int(round(depth_pixel[1]))
        h, w = depth_image.shape
        if not (0 <= u < w and 0 <= v < h):
            return None
        r = max(0, int(radius))
        win = depth_image[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)]
        valid = win[win > 0]
        if valid.size == 0:
            return None
        dis = float(np.median(valid)) * self.depth_scale
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, [float(u), float(v)], dis)
        return dis, int(valid.size), [float(point[0]), float(point[1]), float(point[2])]

    def process_frame(self,depth_pixel, radius=5):
        """
        获取并处理一帧图像，计算圆形区域的平均深度，并显示图像
        :param radius:      圆形区域的半径
        :param center_x:    圆形区域的中心 x 坐标
        :param center_y:    圆形区域的中心 y 坐标
        :return:            圆形区域的平均深度
        """
        # 获取一帧图像
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame= self.get_aligned_images()

        # 计算圆形区域的平均深度
        average_depth = self.calculate_average_depth_in_circle(depth_image, depth_pixel[0], depth_pixel[1], radius)

            # 计算空间坐标
        _,X, Y, Z = self.get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin)

        # 打印结果
        print(f"圆形区域的平均深度值：{average_depth} 毫米,空间坐标: X = {X:.2f}m, Y = {Y:.2f}m, Z = {Z:.2f}m")
        # # 在RGB图像上绘制圆形区域
        # cv2.circle(color_image, (int(depth_pixel[0]), int(depth_pixel[1])), radius, (0, 255, 0), 2)  # 绿色圆形，线宽为2

        # # 显示RGB图像
        # cv2.imshow("Color Image", color_image)
        # cv2.waitKey(2000)  # 等待1毫秒，立即返回，不会阻塞程序
        # cv2.destroyAllWindows()  # 关闭窗口 
        # # 返回平均深度，RGB图像，深度图像和空间坐标
        return average_depth, color_image, depth_image, X, Y, Z

    def get_cam_xyz(self, depth_pixel, aligned_depth_frame, depth_intrin,radius=10):
        
        self.get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin)

    def stop_pipeline(self):
        """停止管道"""
        self.pipeline.stop()
        cv2.destroyAllWindows()

# 主程序入口
def main():
    # 初始化 RealSense 处理器
    realsense_processor = RealSenseProcessor()

    while True:
        # 调用处理函数，获取并显示图像以及计算深度
        average_depth = realsense_processor.process_frame([325, 306])  # [325, 306]目标像素坐标

        # 按'q'键退出
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # 停止管道
    realsense_processor.stop_pipeline()

# 确保main()函数在脚本运行时执行
if __name__ == "__main__":
    main()
