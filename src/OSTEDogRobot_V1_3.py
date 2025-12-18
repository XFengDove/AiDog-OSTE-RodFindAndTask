import datetime
import asyncio
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import cv2
import pyrealsense2 as rs
from DeepCamera import RealSenseUtils


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
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # 异步方式启动摄像头
        await self.loop.run_in_executor(None, self.pipeline.start, config)
        self.is_running = True

    def draw_roi_list(self, img: np.ndarray, rois: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """在图像上绘制ROI区域"""
        for roi in rois:
            x, y, w, h = roi
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img

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

    async def measure_roi_distances(
            self,
            rois: Dict[str, Tuple[int, int, int, int]],
            with_overlay: bool = True,
            with_display: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        异步测量指定ROI区域的距离

        参数:
            rois: ROI区域字典，格式为 {"区域名": (x, y, width, height)}
            with_overlay: 是否生成深度叠加图像
            with_display: 是否显示图像（仅当with_overlay为True时有效）

        返回:
            距离信息字典，格式为 {
                "区域名": {
                    "distances": {"min": 值, "max": 值, "average": 值},
                    "roi": (x, y, width, height),
                    "timestamp": 时间戳
                }
            }
        """
        if not self.is_running:
            await self.initialize_camera()

        # 获取帧数据
        frame_data = await self.get_frame_data()
        if not frame_data:
            return {}

        depth_image, self.color_image = frame_data
        height, width = depth_image.shape

        # 处理结果字典
        results = {}

        # 创建深度叠加图像（如果需要）
        overlay = None
        if with_overlay:
            overlay = await self.loop.run_in_executor(
                None, RealSenseUtils.create_depth_overlay, self.color_image, depth_image
            )

        # 测量每个ROI的距离
        for roi_name, roi in rois.items():
            # 异步执行距离测量
            roi_distance = await self.loop.run_in_executor(
                None, RealSenseUtils.measure_distance_in_roi, depth_image, roi
            )

            # 构建结果
            results[roi_name] = {
                "distances": roi_distance if roi_distance else {},
                "roi": roi,
                "timestamp": datetime.datetime.now()
            }

            # 在叠加图像上绘制ROI和距离信息
            if with_overlay and overlay is not None:
                # 绘制ROI区域
                x, y, w, h = roi
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 显示距离信息
                if roi_distance:
                    y_offset = y + 20
                    for i, (key, value) in enumerate(roi_distance.items()):
                        cv2.putText(
                            overlay,
                            f"{key}: {value:.2f}m",
                            (x, y_offset + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )

        # 添加时间戳
        if with_overlay and overlay is not None:
            cv2.putText(
                overlay,
                f"Time: {datetime.datetime.now().strftime('%H:%M:%S.%f')}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # 显示图像
        if with_display and with_overlay and overlay is not None:
            cv2.imshow("Depth Overlay", overlay)
            #cv2.imshow("Color Image", self.color_image)
            cv2.waitKey(1)

        return results

    async def measure_distances_stream(
            self,
            rois: Dict[str, Tuple[int, int, int, int]],
            interval: float = 0.1,
            max_frames: Optional[int] = None
    ):
        """
        异步生成器，持续测量距离

        参数:
            rois: ROI区域字典
            interval: 测量间隔（秒）
            max_frames: 最大帧数（None表示无限）

        返回:
            异步生成器，每次产生距离信息
        """
        frame_count = 0

        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    break

                distances = await self.measure_roi_distances(
                    rois,
                    with_overlay=True,
                    with_display=True
                )

                if distances:
                    yield distances
                    frame_count += 1

                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            print("距离测量流被取消")
        except Exception as e:
            print(f"距离测量流出错: {e}")

    async def close(self):
        """关闭摄像头"""
        if self.pipeline and self.is_running:
            await self.loop.run_in_executor(None, self.pipeline.stop)
            self.is_running = False
            print("摄像头已关闭")


async def measure_distances_once(
        rois: Dict[str, Tuple[int, int, int, int]],
        display: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    单次测量ROI距离

    参数:
        rois: ROI区域字典
        display: 是否显示图像

    返回:
        距离字典
    """
    processor = AsyncCameraProcessor()

    try:
        distances = await processor.measure_roi_distances(
            rois,
            with_overlay=True,
            with_display=display
        )
        return distances
    finally:
        await processor.close()


async def measure_distances_continuous(
        rois: Dict[str, Tuple[int, int, int, int]],
        callback: callable,
        interval: float = 1/60,
        duration: Optional[float] = None
):
    """
    持续测量ROI距离的便捷函数

    参数:
        rois: ROI区域字典
        callback: 回调函数，接收距离信息作为参数
        interval: 测量间隔（秒）
        duration: 持续时间（秒），None表示无限
    """
    processor = AsyncCameraProcessor()

    try:
        start_time = asyncio.get_event_loop().time()

        async for distances in processor.measure_distances_stream(
                rois,
                interval=interval
        ):
            await callback(distances)

            if duration and (asyncio.get_event_loop().time() - start_time >= duration):
                break

    finally:
        await processor.close()


class RoiSelector:
    """鼠标选择额外Roi测绘区域"""

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


# 使用示例
async def main_example():
    """使用示例"""
    # 定义ROI区域（您可以根据需要修改）
    rois = {
        "Left": (0, 215, 100, 50),
        "Right": (540, 215, 100, 50),
        "Mid": (270, 190, 100, 100),
        "MidDown": (270, 430, 100, 50),
        "MidUp": (270, 0, 100, 50),
    }

    print("=== 单次测量示例 ===")
    distances = await measure_distances_once(rois, display=False)

    for region, data in distances.items():
        print(f"{region}:")
        for dist_type, value in data["distances"].items():
            print(f"  {dist_type}: {value:.2f}m")

    print("\n=== 持续测量示例（持续3秒）===")

    async def print_distances(dist_info):
        for region, data in dist_info.items():
            avg_dist = data["distances"].get("average", 0)
            print(f"{region}: {avg_dist:.2f}m", end=" | ")
        print()

    await measure_distances_continuous(
        rois,
        print_distances,
        interval=1/60,
        duration=3
    )

    processor = AsyncCameraProcessor()

    try:
        # 持续测量并显示
        async for distances in processor.measure_distances_stream(rois, interval=1/60):
            print(f"时间: {distances['Left']['timestamp'].strftime('%H:%M:%S.%f')}")

            # 处理距离数据...
            for region, data in distances.items():
                min_dist = data["distances"].get("min", 0)
                if region == "MidDown":
                    min_dist += 0.05
                if min_dist < 0.5:  # 安全距离检查
                    print(f"警告: {region} 距离过近: {min_dist:.2f}m",end=" | \n")

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        await processor.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    asyncio.run(main_example())