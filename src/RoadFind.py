import cv2
import numpy as np
import math
import requests
import asyncio
from collections import deque
from typing import Optional, Tuple, List
import time

# 从Ysc文件导入异步机器人
try:
    from Ysc_Lite_V1_2 import Ysc_Lite3_Robot
except ImportError:
    # 如果在同一目录下，直接引用
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Ysc_Lite_V1_2 import Ysc_Lite3_Robot


def ReadList_LastData(List: list, ReadLen: int) -> list:
    """读取列表中最后N个数据"""
    return [List[-i] for i in range(1, ReadLen + 1)]


class AsyncRobotController:
    """异步机器人控制器"""

    def __init__(self):
        self.robot = Ysc_Lite3_Robot()
        self.is_moving = False
        self.stop_requested = False
        self.current_action = None
        self.action_queue = deque()
        self.loop = asyncio.get_event_loop()

    async def start(self):
        """启动机器人连接"""
        # 启动心跳检测
        self.robot.start_heart_checking()
        await asyncio.sleep(0.5)  # 等待连接建立

    async def stop(self):
        """停止机器人并清理资源"""
        self.stop_requested = True

        # 停止所有动作
        if self.current_action and not self.current_action.done():
            self.current_action.cancel()

        # 清空队列
        self.action_queue.clear()

        # 发送停止指令
        await self.robot.stop()

        # 关闭连接
        await self.robot.close()

    async def add_action(self, action_type: str, duration: float = 0.1):
        """添加动作到队列并执行"""
        self.is_moving = True

        try:
            if action_type == "left":
                await self.robot.move_left(duration)
            elif action_type == "right":
                await self.robot.move_right(duration)
            elif action_type == "turn_left":
                await self.robot.turn_left(duration)
            elif action_type == "turn_right":
                await self.robot.turn_right(duration)
            elif action_type == "forward":
                await self.robot.move_forward(duration)
            elif action_type == "backward":
                await self.robot.move_backward(duration)
            elif action_type == "turn_left_90":
                await self.robot.turn_left_90_degree()
            elif action_type == "turn_right_90":
                await self.robot.turn_right_90_degree()
            elif action_type == "turn_180":
                await self.robot.turn_around_180_degree()

        except asyncio.CancelledError:
            print(f"动作 {action_type} 被取消")
        finally:
            self.is_moving = False

    def add_action_to_queue(self, action_type: str, duration: float = 0.1):
        """添加动作到队列（非阻塞）"""
        self.action_queue.append((action_type, duration))

    async def process_action_queue(self):
        """处理动作队列"""
        while not self.stop_requested:
            if self.action_queue:
                action_type, duration = self.action_queue.popleft()
                await self.add_action(action_type, duration)
            await asyncio.sleep(0.01)


class AsyncImageProcessor:
    """异步图像处理器"""

    def __init__(self, rioHei: int = 140):
        self.SERVER_URL = "http://172.17.0.1:5000/frame"
        self.ANGLE_THRESHOLD = 5
        self.POSITION_THRESHOLD = 25 + 40
        self.ANGLE_STABLE_THRESHOLD = 10
        self.ANGLE_UPDATE_INTERVAL = 0.05

        # 默认参数
        self.h_min = 20
        self.h_max = 40
        self.s_min = 20
        self.s_max = 255
        self.v_min = 60
        self.v_max = 255
        self.morph_size = 5
        self.fill_area = 500
        self.roi_height = rioHei
        self.centerline_length = self.roi_height * 0.8

        # 分析结果
        self.analysis_result = {
            'action': "前进",
            'angle': 0,
            'position_offset': 0,
            'action_code': "forward",
            'road_centerline': None,
            'road_angle': 0,
            'angle_stable': False,
            'last_angle_update': 0
        }

        # 图像缓存
        self.latest_frame = None
        self.frame_lock = asyncio.Lock()

        # 启动异步图像获取任务
        self.frame_task = asyncio.create_task(self.frame_fetcher_web())

    async def frame_fetcher_web(self):
        """异步从服务器获取图像帧"""
        import aiohttp
        from aiohttp import ClientTimeout

        timeout = ClientTimeout(total=1.0)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                try:
                    async with session.get(self.SERVER_URL) as response:
                        if response.status == 200:
                            data = await response.read()
                            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                async with self.frame_lock:
                                    self.latest_frame = frame
                        await asyncio.sleep(0.03)
                except Exception as e:
                    print(f"获取帧错误: {e}")
                    await asyncio.sleep(0.1)

    async def get_frame(self) -> Optional[np.ndarray]:
        """获取最新的图像帧"""
        async with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def calculate_road_centerline(self, contour, roi_width, roi_height) -> Tuple[Optional[Tuple], float]:
        """计算道路中轴线及其角度"""
        top_points = []
        bottom_points = []

        for point in contour[:, 0, :]:
            if point[1] < roi_height * 0.3:
                top_points.append(point)
            elif point[1] > roi_height * 0.7:
                bottom_points.append(point)

        if not top_points or not bottom_points:
            return None, 0

        top_center = np.mean(top_points, axis=0).astype(int)
        bottom_center = np.mean(bottom_points, axis=0).astype(int)

        dx = bottom_center[0] - top_center[0]
        dy = bottom_center[1] - top_center[1]
        angle = math.degrees(math.atan2(dx, dy))

        center_x = roi_width // 2
        center_y = roi_height // 2

        direction_x = math.sin(math.radians(angle))
        direction_y = math.cos(math.radians(angle))

        start_point = (int(center_x - direction_x * self.centerline_length),
                       int(center_y - direction_y * self.centerline_length))
        end_point = (int(center_x + direction_x * self.centerline_length),
                     int(center_y + direction_y * self.centerline_length))

        return (start_point, end_point), angle

    async def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """异步处理图像帧"""
        if frame is None:
            return None, None

        height, width = frame.shape[:2]
        roi_y = height - self.roi_height
        roi = frame[roi_y:height, :]
        roi_height, roi_width = roi.shape[:2]

        # HSV转换和阈值处理
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((self.morph_size, self.morph_size), np.uint8)
        mask_processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_x = roi_width // 2

        roi_draw = roi.copy()

        if contours:
            main_contour = max(contours, key=cv2.contourArea)

            # 计算道路中轴线和角度
            current_time = time.time()
            if current_time - self.analysis_result['last_angle_update'] > self.ANGLE_UPDATE_INTERVAL:
                centerline, angle = self.calculate_road_centerline(main_contour, roi_width, roi_height)
                self.analysis_result['road_centerline'] = centerline
                self.analysis_result['road_angle'] = angle
                self.analysis_result['angle'] = angle
                self.analysis_result['last_angle_update'] = current_time
                self.analysis_result['angle_stable'] = abs(angle) < self.ANGLE_STABLE_THRESHOLD + 1

            # 绘制中轴线
            if self.analysis_result['road_centerline']:
                cv2.line(roi_draw, self.analysis_result['road_centerline'][0],
                         self.analysis_result['road_centerline'][1], (255, 0, 255), 3)

            # 绘制轮廓和边界框
            cv2.drawContours(roi_draw, [main_contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(main_contour)
            position_offset = (x + w // 2) - center_x
            self.analysis_result['position_offset'] = position_offset

            cv2.rectangle(roi_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(roi_draw, (x + w // 2, 0), (x + w // 2, self.roi_height), (0, 0, 255), 2)
            cv2.line(roi_draw, (center_x, 0), (center_x, self.roi_height), (255, 255, 0), 2)

            # 方向判断
            if position_offset > self.POSITION_THRESHOLD:
                self.analysis_result['action'] = "向右平移"
                self.analysis_result['action_code'] = "right"
            elif position_offset < -self.POSITION_THRESHOLD + 40:
                self.analysis_result['action'] = "向左平移"
                self.analysis_result['action_code'] = "left"
            else:
                self.analysis_result['action'] = "前进"
                self.analysis_result['action_code'] = "forward"

        frame[roi_y:height, :] = roi_draw

        # 显示信息
        cv2.putText(frame, f"Action: {self.analysis_result['action']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Offset: {self.analysis_result['position_offset']} px", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Angle: {self.analysis_result['angle']:.1f}°", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 创建处理结果视图
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_processed_colored = cv2.cvtColor(mask_processed, cv2.COLOR_GRAY2BGR)
        processing_views = np.hstack([
            cv2.resize(frame, (320, 240)),
            cv2.resize(mask_colored, (320, 240)),
            cv2.resize(mask_processed_colored, (320, 240))
        ])

        return frame, processing_views

    def get_analysis_result(self):
        return self.analysis_result.copy()

    def close(self):
        """关闭资源"""
        if self.frame_task and not self.frame_task.done():
            self.frame_task.cancel()


async def calibrate_position_async(WaitTime: float = 1, Rio: int = 140):
    """
    异步位置校准程序
    当位置偏移量和角度在指定时间内持续处于允许范围内时停止
    """
    robot = AsyncRobotController()
    processor = AsyncImageProcessor(rioHei=Rio)
    DegWait = 10

    # 启动机器人
    await robot.start()

    # 启动动作队列处理
    queue_task = asyncio.create_task(robot.process_action_queue())

    # 用于跟踪偏移量和角度在允许范围内的时间
    stable_start_time = None
    CALIBRATION_STABLE_TIME = WaitTime
    last_turn_time = 0
    TURN_COOLDOWN = 1.0

    # 角度历史记录
    angle_history = []
    Angle_History_All = []
    Angle_TF = None

    try:
        while True:
            start_time = time.time()

            # 获取并处理帧
            frame = await processor.get_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            # 处理图像
            frame, processing_views = await processor.process_frame(frame)
            analysis_result = processor.get_analysis_result()

            # 显示结果
            cv2.imshow('Road Detection', frame)
            cv2.imshow('Processing Views', processing_views)

            current_offset = analysis_result['position_offset']
            current_angle = analysis_result['angle']
            current_angle = round(current_angle, 2)

            # 更新角度历史
            if len(angle_history) < 10:
                angle_history.append(current_angle)
            else:
                angle_history = angle_history[1:] + [current_angle]

            Angle_History_All.append(current_angle)

            # 计算平滑角度
            if angle_history:
                smoothed_angle = sum(angle_history) / len(angle_history)
            else:
                smoothed_angle = current_angle

            print(f"<Deg>: {smoothed_angle}")
            # 运动控制
            print(f"<Action>: {analysis_result}")
            if not robot.is_moving:
                action_code = analysis_result['action_code']

                # 清除队列中的旧动作
                robot.action_queue.clear()

                # 根据分析结果添加新动作
                if action_code in ["left", "right", "forward","turn_left", "turn_right"]:
                    robot.add_action_to_queue(action_code, duration=0.1)

                # 角度过大且冷却时间已过才执行旋转
                current_time = time.time()
                if (abs(smoothed_angle) > DegWait and
                        current_time - last_turn_time > TURN_COOLDOWN):

                    rotation_duration = min(0.1, abs(smoothed_angle) / 2000.0)

                    if smoothed_angle > 0:
                        robot.add_action_to_queue("turn_left", rotation_duration)
                    else:
                        robot.add_action_to_queue("turn_right", rotation_duration)

                    last_turn_time = current_time

            # 检查稳定性
            angle_stable = abs(smoothed_angle) < DegWait if not Angle_TF else True
            position_stable = abs(current_offset) < processor.POSITION_THRESHOLD

            if angle_stable:
                Angle_TF = True

            if position_stable and angle_stable:
                if stable_start_time is None:
                    stable_start_time = time.time()
                elif time.time() - stable_start_time >= CALIBRATION_STABLE_TIME:
                    print(f"位置校准完成！偏移量: {abs(current_offset):.1f}px, 角度: {smoothed_angle:.1f}°")
                    return [True, abs(current_offset), smoothed_angle]
            else:
                stable_start_time = None

            # 显示处理信息
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            print(f"处理时间: {processing_time:.2f}s, FPS: {fps:.1f}, "
                  f"偏移量: {current_offset}px, 角度: {smoothed_angle:.1f}°")

            # 按键检测
            key = cv2.waitKey(1)
            if key == ord('s'):
                print(f"当前参数: HSV={[processor.h_min, processor.h_max]}, "
                      f"{[processor.s_min, processor.s_max]}, {[processor.v_min, processor.v_max]}")
            elif key == 27:  # ESC
                print("用户中断校准过程")
                break

            await asyncio.sleep(0.01)  # 让出控制权

    except asyncio.CancelledError:
        print("校准任务被取消")
    except Exception as e:
        print(f"校准过程中发生错误: {e}")

def calibrate_position(WaitTime: float = 1, Rio: int = 140):
    """
    同步包装函数，用于向后兼容
    """
    return asyncio.run(calibrate_position_async(WaitTime, Rio))


# 测试代码
if __name__ == "__main__":
    result = calibrate_position()
    print(f"校准结果: 成功={result[0]}, 偏移量={result[1]}, 角度={result[2]}")