import datetime
from DeepCamera import RealSenseUtils
import pyrealsense2 as rs
import numpy as np
import cv2
import asyncio
from typing import Dict, Tuple, Any, Optional


class AsyncCameraProcessor:
    def __init__(self):
        """初始化摄像头处理器"""
        self.pipeline = None
        self.is_running = False
        self.loop = asyncio.get_event_loop()

    async def initialize_camera(self):
        """初始化RealSense摄像头"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 异步方式启动摄像头
        await self.loop.run_in_executor(None, self.pipeline.start, config)
        self.is_running = True

    def create_roi_list(self, depth_img: np.ndarray, overlay_img: np.ndarray,
                        rois: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, np.ndarray]:
        """在图像上绘制ROI区域"""
        for roi_name, roi in rois.items():
            cv2.rectangle(overlay_img,
                          (roi[0], roi[1]),
                          (roi[0] + roi[2], roi[1] + roi[3]),
                          (0, 255, 0), 2)
        return overlay_img

    async def get_frame_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """异步获取帧数据"""
        if not self.pipeline or not self.is_running:
            return None

        try:
            # 异步等待帧数据
            frames = await self.loop.run_in_executor(None, self.pipeline.wait_for_frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None

            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return depth_image, color_image

        except Exception as e:
            print(f"获取帧数据时出错: {e}")
            return None

    async def measure_roi_distances(self, rois: Dict[str, Tuple[int, int, int, int]],
                                    with_display: bool = False) -> Dict[str, Dict[str, float]]:
        """
        异步测量指定ROI区域的距离

        参数:
            rois: ROI区域字典，格式为 {"区域名": (x, y, width, height)}
            with_display: 是否同时返回显示图像

        返回:
            距离信息字典，格式为 {"区域名": {"min": 值, "max": 值, "average": 值}}
        """
        if not self.is_running:
            await self.initialize_camera()

        # 获取帧数据
        frame_data = await self.get_frame_data()
        if not frame_data:
            return {}

        depth_image, color_image = frame_data

        # 创建深度叠加图像
        overlay = await self.loop.run_in_executor(
            None, RealSenseUtils.create_depth_overlay, color_image, depth_image
        )

        # 绘制ROI区域
        if with_display:
            overlay = self.create_roi_list(depth_image, overlay, rois)

        # 测量每个ROI的距离
        distance_info = {}
        for roi_name, roi in rois.items():
            # 异步执行距离测量
            roi_distance = await self.loop.run_in_executor(
                None, RealSenseUtils.measure_distance_in_roi, depth_image, roi
            )
            distance_info[roi_name] = roi_distance

            # 在图像上显示距离信息
            if with_display and roi_distance:
                y_offset = roi[1] + 10
                for i, (key, value) in enumerate(roi_distance.items()):
                    cv2.putText(overlay, f"{key}:{round(value, 2)}",
                                (roi[0], y_offset + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 添加时间戳
        if with_display:
            cv2.putText(overlay, f"Time: {datetime.datetime.now().strftime('%H:%M:%S.%f')}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 如果需要显示，返回距离信息和图像
        if with_display:
            return {
                "distances": distance_info,
                "overlay": overlay,
                "color_image": color_image,
                "timestamp": datetime.datetime.now()
            }
        else:
            return distance_info

    async def measure_distances_with_callback(self, rois: Dict[str, Tuple[int, int, int, int]],
                                              callback: callable,
                                              interval: float = 0.1):
        """
        持续测量距离并调用回调函数

        参数:
            rois: ROI区域字典
            callback: 回调函数，接收距离信息作为参数
            interval: 测量间隔（秒）
        """
        try:
            while self.is_running:
                distance_info = await self.measure_roi_distances(rois, with_display=False)
                if distance_info:
                    await callback(distance_info)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            print("距离测量任务被取消")
        except Exception as e:
            print(f"距离测量出错: {e}")

    async def close(self):
        """关闭摄像头"""
        if self.pipeline and self.is_running:
            await self.loop.run_in_executor(None, self.pipeline.stop)
            self.is_running = False
            print("摄像头已关闭")


class RoiSelector:
    """ROI选择器类"""

    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.roi = None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.end_point = (x, y)

            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                self.roi = (
                    min(x1, x2),
                    min(y1, y2),
                    abs(x2 - x1),
                    abs(y2 - y1)
                )


async def get_camera_info_async():
    """异步获取摄像头信息"""
    loop = asyncio.get_event_loop()
    camera_info = await loop.run_in_executor(None, RealSenseUtils.get_camera_info)
    return camera_info


async def main_demo():
    """演示如何使用异步摄像头处理器"""
    # 获取摄像头信息
    print("正在获取摄像头信息...")
    camera_info = await get_camera_info_async()
    for info in camera_info:
        print(f"摄像头: {info['name']}")
        print(f"  序列号: {info['serial']}")
        print(f"  固件版本: {info['firmware']}")

    # 创建处理器
    processor = AsyncCameraProcessor()

    try:
        rois = {
            "Left": (0, 215, 100, 50),  # (x, y, width, height)
            "Right": (540, 215, 100, 50),
            "Mid": (270, 190, 100, 100),
            "MidDown": (270, 430, 100, 50),
        }

        # 单次测量示例
        print("\n单次测量示例:")
        distances = await processor.measure_roi_distances(rois, with_display=False)
        for region, data in distances.items():
            print(f"{region}: {data}")

        # 持续测量示例（运行5秒）
        print("\n持续测量示例（5秒）:")

        async def print_distances(dist_info):
            for region, data in dist_info.items():
                print(f"{region}: 平均距离={data.get('average', 0):.2f}m", end=" | ")
            print()

        # 创建持续测量任务
        measurement_task = asyncio.create_task(
            processor.measure_distances_with_callback(rois, print_distances, interval=0.5)
        )

        # 运行5秒
        await asyncio.sleep(5)
        measurement_task.cancel()

        # 带显示的测量
        print("\n带显示的测量（按'q'退出）:")
        while True:
            result = await processor.measure_roi_distances(rois, with_display=True)

            if "overlay" in result:
                cv2.imshow("Depth Overlay", result["overlay"])
                cv2.imshow("Color Image", result["color_image"])

            # 打印距离信息
            for region, data in result.get("distances", {}).items():
                print(f"{region}: {data}")

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        await processor.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 运行演示
    asyncio.run(main_demo())