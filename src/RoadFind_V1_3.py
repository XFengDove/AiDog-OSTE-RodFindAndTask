import numpy as np
import cv2
import math
import time
import asyncio
from typing import Dict, Tuple, Any, Optional, Union


class AsyncRoadDetector:
    """异步道路检测器 - 仅计算角度和偏移量"""

    def __init__(self, roi_height: int = 80, type: str = "Yellow"):
        """
        初始化道路检测器

        参数:
            roi_height: ROI区域高度（像素）
        """
        self.roi_height = roi_height
        self.centerline_length = self.roi_height * 0.8

        self.type = type
        HSV_COLOR = {
            "h_min": 70, "h_max": 150,
            "s_min": 120, "s_max": 255,
            "v_min": 140, "v_max": 255
        }

        # 默认HSV参数（黄色道路）

        if type == "Yellow":
            self.h_min = 20
            self.h_max = 40
            self.s_min = 20
            self.s_max = 250
            self.v_min = 60
            self.v_max = 240
        elif type == "Blue":
            self.h_min = HSV_COLOR["h_min"]
            self.h_max = HSV_COLOR["h_max"]
            self.s_min = HSV_COLOR["s_min"]
            self.s_max = HSV_COLOR["s_max"]
            self.v_min = HSV_COLOR["v_min"]
            self.v_max = HSV_COLOR["v_max"]

        # 形态学处理参数
        self.morph_size = 5

        # 分析结果缓存
        self.analysis_result = {
            'angle': 0,
            'position_offset': 0,
            'road_centerline': None,
            'road_angle': 0,
            'last_angle_update': 0
        }

        # 阈值参数
        self.POSITION_THRESHOLD = 25 + 40
        self.ANGLE_UPDATE_INTERVAL = 0.05

        # 事件循环引用
        self.loop = asyncio.get_event_loop()

    def calculate_road_centerline(self, contour, roi_width, roi_height) -> Tuple[Optional[Tuple], float]:
        """计算道路中轴线及其相对于垂直方向（y轴）的角度"""
        if len(contour) < 3:  # 轮廓点太少无法计算
            return None, 0

        # 获取轮廓的所有点
        points = contour[:, 0, :]

        # 1. 提取关键点：顶部、底部和左右边缘
        # 按y坐标排序
        sorted_by_y = points[np.argsort(points[:, 1])]

        # 顶部区域点（最上面的20%）
        top_y_threshold = roi_height * 0.2
        top_points = points[points[:, 1] < top_y_threshold]

        # 底部区域点（最下面的20%）
        bottom_y_threshold = roi_height * 0.8
        bottom_points = points[points[:, 1] > bottom_y_threshold]

        # 如果没有足够的点，使用整个轮廓
        if len(top_points) < 2 or len(bottom_points) < 2:
            # 使用顶部和底部的最极端点
            if len(sorted_by_y) > 0:
                top_point = sorted_by_y[0]  # 最高点（y最小）
                bottom_point = sorted_by_y[-1]  # 最低点（y最大）
                top_points = np.array([top_point])
                bottom_points = np.array([bottom_point])
            else:
                return None, 0

        # 2. 计算顶部和底部的中心线
        top_center = np.mean(top_points, axis=0).astype(float)
        bottom_center = np.mean(bottom_points, axis=0).astype(float)

        # 3. 计算相对于垂直方向（y轴）的角度
        # 计算x和y方向的差异
        dx = bottom_center[0] - top_center[0]  # x方向变化
        dy = bottom_center[1] - top_center[1]  # y方向变化

        # 避免除零错误
        if abs(dy) < 1e-6:
            # 如果dy接近0，道路几乎是水平的
            if dx > 0:
                angle = 90.0  # 向右水平
            else:
                angle = -90.0  # 向左水平
        else:
            # 计算相对于垂直方向的角度
            # atan2(dx, dy) 给出相对于垂直方向的角度
            # dx为正时向右偏转（正角度），dx为负时向左偏转（负角度）
            angle = math.degrees(math.atan2(dx, dy))

        # 4. 计算中轴线（基于实际检测到的道路中心）
        # 计算道路中心线的实际中点
        road_center_x = (top_center[0] + bottom_center[0]) / 2
        road_center_y = (top_center[1] + bottom_center[1]) / 2

        # 计算中轴线的方向向量（基于角度）
        # 注意：这里使用相对于垂直方向的角度
        direction_x = math.sin(math.radians(angle))
        direction_y = math.cos(math.radians(angle))

        # 计算中轴线的端点
        start_point = (
            int(road_center_x - direction_x * self.centerline_length / 2),
            int(road_center_y - direction_y * self.centerline_length / 2)
        )
        end_point = (
            int(road_center_x + direction_x * self.centerline_length / 2),
            int(road_center_y + direction_y * self.centerline_length / 2)
        )

        # 5. 特殊处理：如果道路边缘超出ROI范围
        # 查找左右边缘的极端点
        if len(points) > 0:
            leftmost = points[np.argmin(points[:, 0])]
            rightmost = points[np.argmax(points[:, 0])]

            # 检查是否超出ROI边界
            roi_left_bound = 0
            roi_right_bound = roi_width - 1
            roi_top_bound = 0
            roi_bottom_bound = roi_height - 1

            # 如果左边缘超出ROI，调整角度计算
            if leftmost[0] <= roi_left_bound + 5:  # 接近左边界
                # 使用右边缘和底部中心重新计算角度
                if len(bottom_points) > 0:
                    bottom_center = np.mean(bottom_points, axis=0).astype(float)
                    dx = bottom_center[0] - rightmost[0]
                    dy = bottom_center[1] - rightmost[1]
                    if abs(dy) > 1e-6:
                        angle = math.degrees(math.atan2(dx, dy))

            # 如果右边缘超出ROI，调整角度计算
            if rightmost[0] >= roi_right_bound - 5:  # 接近右边界
                # 使用左边缘和底部中心重新计算角度
                if len(bottom_points) > 0:
                    bottom_center = np.mean(bottom_points, axis=0).astype(float)
                    dx = bottom_center[0] - leftmost[0]
                    dy = bottom_center[1] - leftmost[1]
                    if abs(dy) > 1e-6:
                        angle = math.degrees(math.atan2(dx, dy))

        return (start_point, end_point), angle

    async def calculate_road_centerline_async(self, contour, roi_width, roi_height) -> Tuple[Optional[Tuple], float]:
        """异步计算道路中轴线及其角度"""
        return await self.loop.run_in_executor(
            None, self.calculate_road_centerline, contour, roi_width, roi_height
        )

    def update_hsv_parameters(self, h_min=None, h_max=None, s_min=None, s_max=None,
                              v_min=None, v_max=None, morph_size=None):
        """更新HSV阈值参数"""
        if h_min is not None:
            self.h_min = h_min
        if h_max is not None:
            self.h_max = h_max
        if s_min is not None:
            self.s_min = s_min
        if s_max is not None:
            self.s_max = s_max
        if v_min is not None:
            self.v_min = v_min
        if v_max is not None:
            self.v_max = v_max
        if morph_size is not None:
            self.morph_size = max(3, morph_size)
            if self.morph_size % 2 == 0:
                self.morph_size += 1

    def _detect_road_sync(self, frame: np.ndarray) -> Dict[str, Any]:
        """同步版本的道路检测"""
        if frame is None:
            return {'angle': 0, 'position_offset': 0, 'road_detected': False, 'contour_area': 0}

        height, width = frame.shape[:2]
        roi_y = max(0, height - self.roi_height)

        roi = frame[roi_y:height, :]
        roi_height, roi_width = roi.shape[:2]

        # HSV转换和阈值处理
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # 形态学处理
        kernel = np.ones((self.morph_size, self.morph_size), np.uint8)
        mask_processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_x = roi_width // 2

        result = {
            'angle': 0,
            'position_offset': 0,
            'road_detected': False,
            'contour_area': 0
        }

        if contours:
            # 选择最大轮廓
            main_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(main_contour)
            result['contour_area'] = contour_area

            # 过滤掉太小的轮廓
            if contour_area > 100 or self.type == "Blue":
                # 计算道路中轴线和角度
                current_time = time.time()
                if current_time - self.analysis_result['last_angle_update'] > self.ANGLE_UPDATE_INTERVAL:
                    centerline, angle = self.calculate_road_centerline(main_contour, roi_width, roi_height)
                    self.analysis_result['road_centerline'] = centerline
                    self.analysis_result['road_angle'] = angle
                    self.analysis_result['angle'] = angle
                    self.analysis_result['last_angle_update'] = current_time

                # 计算位置偏移
                x, y, w, h = cv2.boundingRect(main_contour)
                Con = (x + w // 2)
                position_offset = Con - center_x
                self.analysis_result['position_offset'] = position_offset

                result.update({
                    'angle': self.analysis_result['angle'],
                    'position_offset': position_offset,
                    'road_detected': True,
                    'bounding_box': (x, y, w, h)
                })

        return result

    async def detect_road(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        检测道路并计算角度和偏移量

        参数:
            frame: 输入图像帧 (BGR格式)

        返回:
            包含角度和偏移量的字典:
            {
                'angle': 道路角度(度),
                'position_offset': 位置偏移(像素),
                'road_detected': 是否检测到道路,
                'contour_area': 轮廓面积(像素)
            }
        """
        if frame is None:
            return {'angle': 0, 'position_offset': 0, 'road_detected': False, 'contour_area': 0}

        # 在单独的线程中执行图像处理
        return await self.loop.run_in_executor(None, self._detect_road_sync, frame)

    def _detect_road_with_visualization_sync(self, frame: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """同步版本的道路检测可视化"""
        result = self._detect_road_sync(frame)
        vis_frame = frame.copy()

        if result['road_detected']:
            height, width = frame.shape[:2]
            roi_y = max(0, height - self.roi_height)
            roi = vis_frame[roi_y:height, :]

            # 绘制中轴线
            if self.analysis_result['road_centerline']:
                centerline = self.analysis_result['road_centerline']
                cv2.line(roi, centerline[0], centerline[1], (255, 0, 255), 3)

            # 绘制轮廓和边界框
            if 'bounding_box' in result:
                x, y, w, h = result['bounding_box']
                cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # 绘制中心线
                cv2.line(roi, (x + w // 2, 0), (x + w // 2, self.roi_height), (0, 0, 255), 2)
                center_x = roi.shape[1] // 2
                cv2.line(roi, (center_x, 0), (center_x, self.roi_height), (255, 255, 0), 2)

            # 添加文字信息
            cv2.putText(vis_frame, f"Angle: {result['angle']:.1f}°", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis_frame, f"Offset: {result['position_offset']} px", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis_frame, f"Area: {result['contour_area']:.0f}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result, vis_frame

    async def detect_road_with_visualization(self, frame: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        异步检测道路并返回可视化结果

        参数:
            frame: 输入图像帧

        返回:
            Tuple[检测结果字典, 可视化图像]
        """
        if frame is None:
            return {'angle': 0, 'position_offset': 0, 'road_detected': False}, np.zeros((480, 640, 3), dtype=np.uint8)
        frame = frame[::,::2]

        return await self.loop.run_in_executor(None, self._detect_road_with_visualization_sync, frame)

    async def get_road_angle(self, frame: np.ndarray) -> float:
        """
        异步快速获取道路角度

        参数:
            frame: 输入图像帧

        返回:
            道路角度(度)
        """
        result = await self.detect_road(frame)
        return result['angle']

    async def get_position_offset(self, frame: np.ndarray) -> int:
        """
        异步快速获取位置偏移

        参数:
            frame: 输入图像帧

        返回:
            位置偏移(像素)
        """
        result = await self.detect_road(frame)
        return result['position_offset']

    async def process_frames_stream(self, frame_generator, callback=None, interval: float = 0.0):
        """
        异步处理图像帧流

        参数:
            frame_generator: 异步图像帧生成器
            callback: 回调函数，接收检测结果 (可选)
            interval: 处理间隔 (秒)

        返回:
            异步生成器，产生检测结果
        """
        try:
            async for frame in frame_generator:
                # 检测道路
                result = await self.detect_road(frame)

                # 调用回调函数
                if callback:
                    await callback(result)

                # 产生结果
                yield result

                # 等待间隔
                if interval > 0:
                    await asyncio.sleep(interval)

        except asyncio.CancelledError:
            print("帧流处理被取消")
        except Exception as e:
            print(f"帧流处理错误: {e}")

    async def monitor_road_continuously(self, frame_generator, callback,
                                        check_interval: float = 0.1,
                                        angle_threshold: float = 10.0,
                                        offset_threshold: int = 30):
        """
        连续监控道路状态

        参数:
            frame_generator: 异步图像帧生成器
            callback: 回调函数，接收(角度, 偏移量, 是否需要调整)
            check_interval: 检查间隔 (秒)
            angle_threshold: 角度调整阈值 (度)
            offset_threshold: 偏移调整阈值 (像素)
        """
        try:
            while True:
                async for frame in frame_generator:
                    result = await self.detect_road(frame)

                    angle = result['angle']
                    offset = result['position_offset']

                    # 判断是否需要调整
                    needs_adjustment = (
                            abs(angle) > angle_threshold or
                            abs(offset) > offset_threshold
                    )

                    # 调用回调
                    await callback(angle, offset, needs_adjustment)

                    await asyncio.sleep(check_interval)

        except asyncio.CancelledError:
            print("道路监控被取消")

    def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前检测状态

        返回:
            当前状态字典
        """
        return self.analysis_result.copy()


class AsyncFrameProvider:
    """异步帧提供器基类（接口示例）"""

    async def get_frame(self) -> Optional[np.ndarray]:
        """
        获取图像帧（需要子类实现）

        返回:
            图像帧或None
        """
        raise NotImplementedError("子类必须实现此方法")

    async def frames_generator(self, max_frames: Optional[int] = None):
        """
        生成图像帧（异步生成器）

        参数:
            max_frames: 最大帧数

        返回:
            异步图像帧生成器
        """
        frame_count = 0
        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    break

                frame = await self.get_frame()
                if frame is not None:
                    yield frame
                    frame_count += 1

                await asyncio.sleep(0.01)  # 约30fps

        except asyncio.CancelledError:
            print("帧生成器被取消")


class MockFrameProvider(AsyncFrameProvider):
    """模拟帧提供器（用于测试）"""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height

    async def get_frame(self) -> Optional[np.ndarray]:
        """生成随机测试图像"""
        await asyncio.sleep(0.033)  # 模拟获取延迟
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)


# 使用示例
async def main_example():
    """异步使用示例"""
    # 创建道路检测器
    detector = AsyncRoadDetector(roi_height=140)

    # 创建模拟帧提供器
    frame_provider = MockFrameProvider()

    print("=== 单帧检测示例 ===")

    # 获取一帧并检测
    frame = await frame_provider.get_frame()
    if frame is not None:
        result = await detector.detect_road(frame)
        print(f"道路角度: {result['angle']:.1f}°")
        print(f"位置偏移: {result['position_offset']}px")
        print(f"检测到道路: {result['road_detected']}")

    print("\n=== 连续处理示例（持续2秒）===")

    async def process_callback(result):
        """处理回调函数"""
        print(f"角度: {result['angle']:.1f}°, 偏移: {result['position_offset']}px")

    # 连续处理2秒
    start_time = time.time()
    try:
        async for result in detector.process_frames_stream(frame_provider.frames_generator(), callback=process_callback,
                                                           interval=0.2):
            print(result)
            if time.time() - start_time > 2:
                break
    except asyncio.CancelledError:
        pass

    print("\n=== 监控模式示例 ===")

    async def monitoring_callback(angle, offset, needs_adjustment):
        """监控回调"""
        status = "需要调整" if needs_adjustment else "正常"
        print(f"角度: {angle:.1f}°, 偏移: {offset}px, 状态: {status}")

    # 创建监控任务
    monitor_task = asyncio.create_task(
        detector.monitor_road_continuously(
            frame_provider.frames_generator(),
            monitoring_callback,
            check_interval=0.3,
            angle_threshold=5.0,
            offset_threshold=20
        )
    )

    # 运行1.5秒后停止
    await asyncio.sleep(1.5)
    monitor_task.cancel()

    try:
        await monitor_task
    except asyncio.CancelledError:
        print("监控任务已取消")


if __name__ == '__main__':
    asyncio.run(main_example())