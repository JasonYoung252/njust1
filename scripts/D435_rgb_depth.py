import time

import pyrealsense2 as rs
import numpy as np
import cv2

from depth_localizer import CameraIntrinsics, LocalizerConfig, robust_roi_depth


class FrameBundle:
    """同一次采集得到的数据：最后一帧彩色图 + 多帧对齐深度（m）+ 彩色内参。"""

    def __init__(self, color_image, depth_stack_m, intrinsics, depth_scale, aligned_depth_frame, color_intrinsics):
        self.color_image = color_image              # (H, W, 3) BGR
        self.depth_stack_m = depth_stack_m          # (N, H, W) float32，单位 m，0 为无效
        self.intrinsics = intrinsics                # CameraIntrinsics（对齐后的深度使用彩色内参）
        self.depth_scale = depth_scale
        self.aligned_depth_frame = aligned_depth_frame  # 最后一帧，兼容原接口
        self.color_intrinsics = color_intrinsics    # rs.intrinsics，兼容 rs2_deproject_pixel_to_point
        self.timestamp = time.time()


class RealSenseProcessor:
    def __init__(self, fps=30, color_size=(1920, 1080), depth_size=(848, 480), visual_preset=None,
                 laser_power=None, emitter_enabled=True, use_filters=True, warmup_frames=30,
                 min_depth_m=0.10, max_depth_m=1.5):
        """
        初始化 RealSense 管道和流设置

        :param fps:            帧率
        :param color_size:     彩色分辨率，OCR 需要高分辨率，默认保持 1920×1080
        :param depth_size:     深度分辨率。原程序用 1280×720，最小测距约 26 cm；
                               848×480 最小测距约 17 cm，30–40 cm 工作距离余量更大
        :param visual_preset:  None 表示不修改；可选 'default' 'high_accuracy' 'high_density' 'medium_density'
        :param laser_power:    None 表示不修改；'max' 表示最大；或给出数值（mW）
        :param emitter_enabled: 是否开启红外结构光发射器
        :param use_filters:    是否启用后处理滤波（阈值、视差域空间滤波；不用时间滤波和空洞填充）
        :param warmup_frames:  启动后丢弃的帧数，等待自动曝光稳定
        """
        # 初始化管道
        self.pipeline = rs.pipeline()

        # 配置流
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, fps)  # RGB流
        self.config.enable_stream(rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, fps)   # 深度流

        # 启动管道
        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self._configure_depth_sensor(depth_sensor, visual_preset, laser_power, emitter_enabled)

        # 对齐对象只创建一次（原程序每帧新建一次）
        self.align = rs.align(rs.stream.color)

        # 后处理滤波：Intel 建议的顺序。不使用时间滤波（机械臂移动后会混入旧视角的深度），
        # 不使用空洞填充（容易产生错误深度）。
        self.filters = []
        self._filter_on_frameset = True
        if use_filters:
            threshold = rs.threshold_filter()
            threshold.set_option(rs.option.min_distance, float(min_depth_m))
            threshold.set_option(rs.option.max_distance, float(max_depth_m))
            self.filters = [threshold, rs.disparity_transform(True), rs.spatial_filter(), rs.disparity_transform(False)]

        # 对齐后的深度与彩色图同分辨率，使用彩色相机内参
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intr = color_profile.get_intrinsics()
        self.fx = self.color_intr.fx
        self.fy = self.color_intr.fy
        self.cx = self.color_intr.ppx  # 主点（cx）
        self.cy = self.color_intr.ppy  # 主点（cy）
        self.intrinsics = CameraIntrinsics(self.fx, self.fy, self.cx, self.cy)

        # 预热：丢弃前若干帧
        for _ in range(int(warmup_frames)):
            self.pipeline.wait_for_frames()

    # ------------------------------------------------------------------------------------------
    # 相机配置与取帧
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _configure_depth_sensor(depth_sensor, visual_preset, laser_power, emitter_enabled):
        try:
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if emitter_enabled else 0.0)
            if laser_power is not None and depth_sensor.supports(rs.option.laser_power):
                rng = depth_sensor.get_option_range(rs.option.laser_power)
                value = rng.max if laser_power == 'max' else min(max(float(laser_power), rng.min), rng.max)
                depth_sensor.set_option(rs.option.laser_power, value)
            if visual_preset and depth_sensor.supports(rs.option.visual_preset):
                preset = getattr(rs.rs400_visual_preset, str(visual_preset))
                depth_sensor.set_option(rs.option.visual_preset, int(preset))
        except Exception as e:  # 个别固件不支持某些选项，不影响主流程
            print(f"深度相机参数设置失败（忽略）：{e}")

    def _apply_filters(self, frame):
        for flt in self.filters:
            frame = flt.process(frame)
        return frame

    def _next_aligned(self):
        """取一帧：先在原始深度上滤波，再对齐到彩色图。"""
        frames = self.pipeline.wait_for_frames()
        if self.filters and self._filter_on_frameset:
            try:
                filtered = self._apply_filters(frames).as_frameset()
                if filtered and filtered.get_depth_frame():
                    frames = filtered
                else:
                    self._filter_on_frameset = False
            except Exception:
                # 部分 SDK 版本不支持对 frameset 直接滤波，改为对齐后再滤波
                self._filter_on_frameset = False
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if self.filters and not self._filter_on_frameset:
            aligned_depth_frame = self._apply_filters(aligned_depth_frame).as_depth_frame()
        return aligned_depth_frame, color_frame

    def capture_bundle(self, n_frames=5, settle_frames=3):
        """
        连续采集 n_frames 帧（机械臂静止时调用）。识别和取深度都用这一次采集的数据。
        先丢弃 settle_frames 帧：管道队列里可能还留着机械臂运动前的旧帧。
        """
        for _ in range(int(settle_frames)):
            self.pipeline.wait_for_frames()
        depth_list = []
        color_image = None
        aligned_depth_frame = None
        for _ in range(max(1, int(n_frames))):
            aligned_depth_frame, color_frame = self._next_aligned()
            if not aligned_depth_frame or not color_frame:
                continue
            depth_raw = np.asanyarray(aligned_depth_frame.get_data())
            depth_list.append(depth_raw.astype(np.float32) * self.depth_scale)
            color_image = np.asanyarray(color_frame.get_data()).copy()
        if not depth_list:
            raise RuntimeError("RealSense 未取到有效帧")
        return FrameBundle(color_image, np.stack(depth_list, axis=0), self.intrinsics, self.depth_scale,
                           aligned_depth_frame, self.color_intr)

    # ------------------------------------------------------------------------------------------
    # 以下为原有接口，保持签名不变
    # ------------------------------------------------------------------------------------------
    def calculate_average_depth_in_circle(self, depth_image, center_x, center_y, radius=10):
        """
        计算以圆心为中心，半径为 radius 的圆形区域内的平均深度（忽略 0 值，单位与输入相同）
        """
        center_x = int(center_x)
        center_y = int(center_y)
        height, width = depth_image.shape
        y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
        x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
        patch = depth_image[y0:y1, x0:x1].astype(np.float64)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2) & (patch != 0)
        if not mask.any():
            return 0
        return float(patch[mask].mean())

    def get_xyz_to_uv(self, x, y, z):
        """
        相机坐标 -> 像素坐标（对齐后的彩色图像素）
        """
        u = (self.fx * x / z) + self.cx  # 计算像素坐标 u
        v = (self.fy * y / z) + self.cy  # 计算像素坐标 v
        pixel = [int(u), int(v)]
        return u, v, pixel

    def get_aligned_images(self):
        """
        从Intel RealSense摄像头获取对齐后的RGB图像和深度图像
        """
        aligned_depth_frame, color_frame = self._next_aligned()

        # 相机参数的获取
        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）

        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        color_image = np.asanyarray(color_frame.get_data())  # RGB图
        # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame

    def get_3d_camera_coordinate(self, depth_pixel, aligned_depth_frame, depth_intrin, radius=6):
        """
        将图像中的某个像素点转换到相机坐标系下的三维空间坐标
        原程序只读一个像素，遇到空洞得到 (0, 0, 0)。现在中心像素无效时，改用邻域有效深度的中位数；
        邻域也没有有效深度时返回 dis = 0，调用方必须检查。
        """
        x = int(depth_pixel[0])
        y = int(depth_pixel[1])
        dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
        if dis <= 0:
            depth_m = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32) * self.depth_scale
            z, _, _, _ = robust_roi_depth(depth_m, (x - radius, y - radius, x + radius, y + radius),
                                          LocalizerConfig(min_roi_valid_ratio=0.05, min_roi_valid_px=5))
            dis = z if z is not None else 0.0
        if dis <= 0:
            print(f'depth_pixel {depth_pixel} 附近没有有效深度')
            return 0.0, 0.0, 0.0, 0.0
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [float(x), float(y)], float(dis))
        return dis, camera_coordinate[0], camera_coordinate[1], camera_coordinate[2]

    def process_frame(self, depth_pixel, radius=5):
        """
        获取并处理一帧图像，返回圆形区域平均深度（mm）、彩色图、深度图和空间坐标（m）
        """
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame = self.get_aligned_images()
        average_depth = self.calculate_average_depth_in_circle(depth_image, depth_pixel[0], depth_pixel[1], radius)
        _, X, Y, Z = self.get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin)
        print(f"圆形区域的平均深度值：{average_depth} 毫米,空间坐标: X = {X:.3f}m, Y = {Y:.3f}m, Z = {Z:.3f}m")
        return average_depth, color_image, depth_image, X, Y, Z

    def get_cam_xyz(self, depth_pixel, aligned_depth_frame, depth_intrin, radius=10):
        return self.get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin)

    def stop_pipeline(self):
        """停止管道"""
        self.pipeline.stop()
        cv2.destroyAllWindows()


# 主程序入口：实时显示中心像素附近的深度有效率，便于现场诊断
def main():
    realsense_processor = RealSenseProcessor()
    cfg = LocalizerConfig()
    try:
        while True:
            bundle = realsense_processor.capture_bundle(n_frames=1, settle_frames=0)
            depth = bundle.depth_stack_m[0]
            h, w = depth.shape
            u, v = w // 2, h // 2
            z, sigma, ratio, n = robust_roi_depth(depth, (u - 20, v - 20, u + 20, v + 20), cfg)
            vis = bundle.color_image.copy()
            cv2.rectangle(vis, (u - 20, v - 20), (u + 20, v + 20), (0, 255, 0), 2)
            text = "valid %.0f%%  z=%s" % (ratio * 100, "%.3f m" % z if z is not None else "invalid")
            cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("D435 depth check", cv2.resize(vis, (w // 2, h // 2)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        realsense_processor.stop_pipeline()


# 确保main()函数在脚本运行时执行
if __name__ == "__main__":
    main()
