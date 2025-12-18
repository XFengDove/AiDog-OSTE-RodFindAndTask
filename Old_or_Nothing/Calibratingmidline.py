import cv2
import numpy as np
import math
import requests
import socket
import struct
import time
import threading
from collections import deque
from enum import IntEnum
import asyncio
from typing import Dict, Tuple, Any, Optional
from sports.ysc_lite3_control import Ysc_Lite3_Robot
from DeepCamera import RealSenseUtils
import pyrealsense2 as rs



class InstructionType(IntEnum):
    SIMPLE = 0
    COMPLEX = 1

def ReadList_LastData(List:list,ReadLen:int)->list:
    return [List[-i] for i in range(1,ReadLen+1)]
class RobotController:
    def __init__(self):
        # UDP socket
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = ('192.168.1.120', 43893)
        self.is_moving = False
        self.last_action_time = time.time()
        self.action_queue = deque()
        self.action_lock = threading.Lock()
        self.stop_requested = False

        # 启动心跳线程
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()

        # 启动动作执行线程
        self.action_thread = threading.Thread(target=self.execute_actions)
        self.action_thread.daemon = True
        self.action_thread.start()

    def send_heartbeat(self):
        """发送心跳包保持连接"""
        while not self.stop_requested:
            data = struct.pack("<3i", 0x21040001, 0, 0)
            self.udp_socket.sendto(data, self.send_addr)
            time.sleep(0.5)

    def send_data(self, code, value, type):
        """发送控制指令"""
        data = struct.pack("<3i", code, value, type)
        self.udp_socket.sendto(data, self.send_addr)

    def add_action(self, action_type, duration=0.1):
        """添加动作到队列"""
        with self.action_lock:
            self.action_queue.append((action_type, duration))

    def execute_actions(self):
        """执行动作队列中的动作"""
        while not self.stop_requested:
            if self.action_queue:
                with self.action_lock:
                    action_type, duration = self.action_queue.popleft()

                self.is_moving = True
                start_time = time.time()

                # 使用固定时间循环代替while循环
                if action_type == "left":
                    self.send_data(0x21010131, -17000, 0)
                elif action_type == "right":
                    self.send_data(0x21010131, 17000, 0)
                elif action_type == "turn_left":
                    self.send_data(0x21010135, -1000, 0)
                elif action_type == "turn_right":
                    self.send_data(0x21010135, 1000, 0)
                elif action_type == "forward":
                    self.send_data(0x21010130, 13000, 0)

                # 等待动作完成
                time.sleep(duration)

                # 发送停止指令
                if action_type in ["left", "right"]:
                    self.send_data(0x21010131, 0, 0)
                elif action_type in ["turn_left", "turn_right"]:
                    self.send_data(0x21010135, 0, 0)
                elif action_type == "forward":
                    self.send_data(0x21010130, 0, 0)

                self.is_moving = False
                self.last_action_time = time.time()
            else:
                time.sleep(0.01)

    def stop(self):
        """停止所有运动并清理资源"""
        self.stop_requested = True
        # 清空动作队列
        with self.action_lock:
            self.action_queue.clear()
        # 发送停止指令
        self.send_data(0x21010130, 0, 0)  # 停止前进
        self.send_data(0x21010131, 0, 0)  # 停止左右移动
        self.send_data(0x21010135, 0, 0)  # 停止旋转
        time.sleep(0.1)  # 确保指令发送完成
        self.udp_socket.close()


class ImageProcessor:
    def __init__(self, rioHei: int = 140):
        self.SERVER_URL = "http://172.17.0.1:5000/frame"
        self.scan_lines = [0.2, 0.4, 0.6]
        self.ANGLE_THRESHOLD = 5
        self.POSITION_THRESHOLD = 25 + 40
        self.ANGLE_STABLE_THRESHOLD = 10  # 角度稳定阈值(度)
        self.ANGLE_UPDATE_INTERVAL = 0.05  # 角度更新间隔(秒)

        # 默认参数
        self.h_min = 20
        self.h_max = 40
        self.s_min = 20
        self.s_max = 255
        self.v_min = 60
        self.v_max = 255
        self.morph_size = 5
        self.fill_area = 500
        self.roi_height = rioHei  # 看眼前100个像素的道路
        self.centerline_length = self.roi_height * 0.8  # 中轴线长度

        # 分析结果
        self.analysis_result = {
            'action': "前进",
            'angle': 0,
            'position_offset': 0,
            'action_code': "1",
            'road_centerline': None,  # 道路中轴线端点
            'road_angle': 0,  # 道路角度(度)
            'angle_stable': False,  # 角度是否稳定
            'last_angle_update': 0  # 上次角度更新时间
        }

        # 初始化调试窗口
        self.init_debug_windows()

        # 图像缓存（初始化摄像头，如果已经初始化，可能报错）
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        # self.camera = cv2.VideoCapture(4)
        # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # self.camera.set(cv2.CAP_PROP_FPS, 30)
        # time.sleep(2)
        # 启动图像获取线程
        self.frame_thread = threading.Thread(target=self.frame_fetcher_Web)
        self.frame_thread.daemon = True
        self.frame_thread.start()

    def frame_fetcher(self):
        # 修改
        """
        截取摄像头本地缓存图像，需禁用send_image，额外初始化摄像头
        :return: None
        """
        while True:
            try:
                success, frame = self.camera.read()
                if success:
                    resized_frame = cv2.resize(frame, (640, 360))
                    with self.frame_lock:
                        self.latest_frame = resized_frame
                time.sleep(0.03)


            except Exception as e:
                print(f"获取帧错误: {e}")
                time.sleep(0.1)

    def init_debug_windows(self):
        """初始化调试窗口和滑块"""
        cv2.namedWindow('Road Detection')
        cv2.namedWindow('Processing Views')

        # 创建控制滑块
        cv2.createTrackbar('H_min', 'Road Detection', self.h_min, 179, self.nothing)
        cv2.createTrackbar('H_max', 'Road Detection', self.h_max, 179, self.nothing)
        cv2.createTrackbar('S_min', 'Road Detection', self.s_min, 255, self.nothing)
        cv2.createTrackbar('S_max', 'Road Detection', self.s_max, 255, self.nothing)
        cv2.createTrackbar('V_min', 'Road Detection', self.v_min, 255, self.nothing)
        cv2.createTrackbar('V_max', 'Road Detection', self.v_max, 255, self.nothing)
        cv2.createTrackbar('Morph Size', 'Road Detection', self.morph_size, 20, self.nothing)
        cv2.createTrackbar('Fill Area', 'Road Detection', self.fill_area, 2000, self.nothing)
        cv2.createTrackbar('ROI Height', 'Road Detection', self.roi_height, 480, self.nothing)

    def nothing(self, x):
        pass

    def frame_fetcher_Web(self):
        """从服务器获取图像帧的线程"""
        while True:
            try:
                response = requests.get(self.SERVER_URL, timeout=0.5)
                if response.status_code == 200:
                    frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.frame_lock:
                            self.latest_frame = frame
                time.sleep(0.03)  # 控制获取频率
            except Exception as e:
                print(f"获取帧错误: {e}")
                time.sleep(0.1)

    def get_frame(self):
        """获取最新的图像帧"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def update_parameters(self):
        """从滑块更新处理参数"""
        self.h_min = cv2.getTrackbarPos('H_min', 'Road Detection')
        self.h_max = cv2.getTrackbarPos('H_max', 'Road Detection')
        self.s_min = cv2.getTrackbarPos('S_min', 'Road Detection')
        self.s_max = cv2.getTrackbarPos('S_max', 'Road Detection')
        self.v_min = cv2.getTrackbarPos('V_min', 'Road Detection')
        self.v_max = cv2.getTrackbarPos('V_max', 'Road Detection')
        self.morph_size = max(3, cv2.getTrackbarPos('Morph Size', 'Road Detection'))
        self.fill_area = cv2.getTrackbarPos('Fill Area', 'Road Detection')
        self.roi_height = cv2.getTrackbarPos('ROI Height', 'Road Detection')

        if self.morph_size % 2 == 0:
            self.morph_size += 1

    def calculate_road_centerline(self, contour, roi_width, roi_height):
        """计算道路中轴线及其角度"""
        # 计算轮廓的上下边缘
        top_points = []
        bottom_points = []
        print(f"{roi_width, roi_height}")

        for point in contour[:, 0, :]:
            if point[1] < roi_height * 0.3:  # 上1/3部分
                top_points.append(point)
            elif point[1] > roi_height * 0.7:  # 下1/3部分
                bottom_points.append(point)

        if not top_points or not bottom_points:
            return None, 0

        # 计算上下边缘的中心点
        top_center = np.mean(top_points, axis=0).astype(int)
        bottom_center = np.mean(bottom_points, axis=0).astype(int)

        # 计算角度 (以垂直方向为0度)
        dx = bottom_center[0] - top_center[0]
        dy = bottom_center[1] - top_center[1]
        angle = math.degrees(math.atan2(dx, dy))

        # 中轴线端点 (可视化)
        center_x = roi_width // 2
        center_y = roi_height // 2

        # 中轴线方向向量
        direction_x = math.sin(math.radians(angle))
        direction_y = math.cos(math.radians(angle))

        # 计算起点和终点
        start_point = (int(center_x - direction_x * self.centerline_length),
                       int(center_y - direction_y * self.centerline_length))
        end_point = (int(center_x + direction_x * self.centerline_length),
                     int(center_y + direction_y * self.centerline_length))

        return (start_point, end_point), angle

    def process_frame(self, frame):
        """处理图像帧并分析道路情况"""
        if frame is None:
            return None, None

        self.update_parameters()
        height, width = frame.shape[:2]
        roi_y = height - self.roi_height
        roi = frame[roi_y:height, :]
        roi_height, roi_width = roi.shape[:2]

        # 简化处理流程
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((self.morph_size, self.morph_size), np.uint8)
        mask_processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_x = roi_width // 2

        # 重置分析结果
        self.analysis_result = {
            'action': "前进",
            'angle': self.analysis_result['angle'],  # 保留上次的角度值
            'position_offset': 0,
            'action_code': "1",
            'road_centerline': self.analysis_result['road_centerline'],  # 保留上次的中轴线
            'road_angle': self.analysis_result['road_angle'],  # 保留上次的角度
            'angle_stable': self.analysis_result['angle_stable'],  # 保留上次的稳定性
            'last_angle_update': self.analysis_result['last_angle_update']  # 保留更新时间
        }

        roi_draw = roi.copy()

        if contours:
            main_contour = max(contours, key=cv2.contourArea)

            # 计算道路中轴线和角度（限制更新频率）
            current_time = time.time()
            if current_time - self.analysis_result['last_angle_update'] > self.ANGLE_UPDATE_INTERVAL:
                centerline, angle = self.calculate_road_centerline(main_contour, roi_width, roi_height)
                self.analysis_result['road_centerline'] = centerline
                self.analysis_result['road_angle'] = angle
                self.analysis_result['angle'] = angle
                self.analysis_result['last_angle_update'] = current_time

                # 计算角度稳定性
                self.analysis_result['angle_stable'] = abs(angle) < self.ANGLE_STABLE_THRESHOLD + 1

            # 绘制中轴线
            centerline = self.analysis_result['road_centerline']
            if centerline:
                cv2.line(roi_draw, centerline[0], centerline[1], (255, 0, 255), 3)

            # 绘制轮廓和边界框
            cv2.drawContours(roi_draw, [main_contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(main_contour)
            position_offset = (x + w // 2) - center_x
            self.analysis_result['position_offset'] = position_offset

            cv2.rectangle(roi_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(roi_draw, (x + w // 2, 0), (x + w // 2, self.roi_height), (0, 0, 255), 2)
            cv2.line(roi_draw, (center_x, 0), (center_x, self.roi_height), (255, 255, 0), 2)

            # 简化的方向判断
            if position_offset > self.POSITION_THRESHOLD:
                self.analysis_result['action'] = "向右平移"
                self.analysis_result['action_code'] = "right"
            elif position_offset < -self.POSITION_THRESHOLD + 40:
                self.analysis_result['action'] = "向左平移"
                self.analysis_result['action_code'] = "left"
            else:
                self.analysis_result['action'] = "前进"

        frame[roi_y:height, :] = roi_draw

        # 显示信息
        cv2.putText(frame, f"Action: {self.analysis_result['action']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Offset: {self.analysis_result['position_offset']} px", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Angle: {self.analysis_result['angle']:.1f}°", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Angle Stable: {self.analysis_result['angle_stable']}", (20, 160),
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
        return self.analysis_result

    def close_windows(self):
        """关闭所有OpenCV窗口"""
        cv2.destroyAllWindows()


def calibrate_position(WaitTime: float = 1, Rio=140):
    """
    执行位置校准程序
    当位置偏移量和角度在2秒内持续处于允许范围内时停止
    """
    robot = RobotController()
    processor = ImageProcessor(rioHei=Rio)
    DegWait = 10

    # 用于跟踪偏移量和角度在允许范围内的时间
    stable_start_time = None
    CALIBRATION_STABLE_TIME = WaitTime  # 需要稳定2秒
    last_processing_time = time.time()

    # 用于角度稳定性判断的变量
    angle_history = [] # 存储最近10次的角度值

    Angle_History_All = [] #全部角度记录
    last_turn_time = 0  # 上次旋转的时间
    TURN_COOLDOWN = 1.0  # 旋转冷却时间(秒)

    Angle_TF=None

    try:
        while True:
            start_time = time.time()

            # 获取并处理帧
            frame = processor.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            frame, processing_views = processor.process_frame(frame)
            analysis_result = processor.get_analysis_result()

            # 显示结果
            cv2.imshow('Road Detection', frame)
            cv2.imshow('Processing Views', processing_views)

            current_offset = analysis_result['position_offset']
            current_angle = analysis_result['angle']
            current_angle = round(current_angle,2)
            if len(angle_history) <=10:
                pass
            else:
                angle_history=ReadList_LastData(angle_history,10)
            if len(angle_history) !=0:
                Average_Deg = sum(angle_history) / len(angle_history)# 平均数
                Median_Deg = np.median(angle_history) # 中位数
                if (Average_Deg > DegWait or
                        len(angle_history)<5 or
                        angle_history[0]==0) or abs(abs(angle_history[0])-abs(angle_history[-1])>=DegWait):
                    # 更新角度历史
                    angle_history.append(current_angle)
                else:
                    pass
            else:
                angle_history.append(current_angle)
                Average_Deg = sum(angle_history) / len(angle_history)  # 平均数
                Median_Deg = np.median(angle_history)  # 中位数
            print(angle_history)
            Angle_History_All.append(round(current_angle, 2))






            # 计算平滑角度(中值滤波)
            if angle_history:
                smoothed_angle =Average_Deg
            else:
                smoothed_angle = current_angle
            print(Average_Deg, Median_Deg, smoothed_angle)
            # 运动控制
            if not robot.is_moving:
                action_code = analysis_result['action_code']
                current_time = time.time()
                with robot.action_lock:
                    robot.action_queue.clear()
                # # 发送停止指令
                robot.send_data(0x21010130, 0, 0)  # 停止前进
                robot.send_data(0x21010131, 0, 0)  # 停止左右移动
                robot.send_data(0x21010135, 0, 0)  # 停止旋转

                # 只有在冷却时间过后才允许旋转
                if action_code in ["left", "right", "forward"]:
                    # 正常执行动作
                    robot.add_action(action_code, duration=0.1)

                # 角度过大且冷却时间已过才执行旋转
                if (abs(smoothed_angle) > DegWait and
                        current_time - last_turn_time > TURN_COOLDOWN):
                    robot_controller = Ysc_Lite3_Robot()
                    rotation_duration = min(0.1, abs(smoothed_angle) / 2000.0)  # 限制最大旋转时间

                    if smoothed_angle > 10:
                        robot_controller.turn_l(rotation_duration)
                    else:
                        robot_controller.turn_r(rotation_duration)

                    last_turn_time = current_time

            # 检查偏移量和角度是否在允许范围内
            angle_stable = abs(smoothed_angle) < DegWait if not Angle_TF else True
            position_stable = abs(current_offset) < processor.POSITION_THRESHOLD
            if angle_stable:
                Angle_TF=True
            if position_stable and angle_stable:
                # 位置和角度都稳定
                if stable_start_time is None:
                    # 开始记录稳定时间
                    stable_start_time = time.time()
                else:
                    # 检查是否已达到要求的稳定时间
                    if time.time() - stable_start_time >= CALIBRATION_STABLE_TIME:
                        print(f"位置校准完成！偏移量: {abs(current_offset):.1f}px, 角度: {smoothed_angle:.1f}°")
                        return [True, abs(current_offset), smoothed_angle]
            else:
                # 位置或角度不稳定，重置稳定计时器
                stable_start_time = None

            # 计算并显示处理时间
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            print(f"处理时间: {processing_time:.2f}s, FPS: {fps:.1f}, "
                  f"偏移量: {current_offset}px, 角度: {smoothed_angle:.1f}°, "
                  f"稳定时间: {time.time() - stable_start_time if stable_start_time else 0:.1f}s")

            # 按键操作
            key = cv2.waitKey(1)
            if key == ord('s'):
                print(f"当前参数: HSV={[processor.h_min, processor.h_max]}, "
                      f"{[processor.s_min, processor.s_max]}, {[processor.v_min, processor.v_max]}")
            elif key == 27:  # ESC键

                print("用户中断校准过程")
                break

    except Exception as e:
        print(f"校准过程中发生错误: {e}")
    finally:
        # 清理资源
        # Pandas = pandas.Series(Angle_History_All)
        print(f"全部角度：{Angle_History_All}\n"
              f"最大值:{round(max(Angle_History_All), 2)}<UNK>\n "
              f"最小值:{round(min(Angle_History_All), 2)}\n"
              f"中位数：{round(np.median(Angle_History_All), 2)}\n"
              f"平均数：{round(sum(Angle_History_All) / len(Angle_History_All), 2)}\n)")
              # f"众数:{round(Pandas.mode(), 2)}\n")
        robot.stop()
        try:
            processor.camera.release()
            cv2.destroyAllWindows()
        except AttributeError:
            pass
        processor.close_windows()

        print("校准程序已停止")


# 测试代码
if __name__ == "__main__":
    result = calibrate_position()
    print(f"校准结果: 成功={result[0]}, 偏移量={result[1]}, 角度={result[2]}")