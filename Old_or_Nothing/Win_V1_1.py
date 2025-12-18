import cv2
import asyncio
from enum import Enum
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import OSTEDogRobot_V1_3 as DeepCamera
import RoadFind_V1_3 as RoadFind
import Ysc_Lite_V1_2 as Lite
import time


class RobotState(Enum):
    """机器狗状态枚举"""
    NORMAL_NAVIGATION = 1  # 正常导航模式
    BLUE_AREA_DETECTED = 2  # 检测到蓝色区域
    PERFORMING_ACTION = 3  # 执行动作中
    RETURN_TO_NORMAL = 4  # 返回正常模式


class StateMachine:
    """状态机管理类"""

    def __init__(self):
        self.current_state = RobotState.NORMAL_NAVIGATION
        self.action_start_time = None
        self.action_duration = 5.0  # 默认动作执行时间（秒）
        self.blue_area_processed = False

    def transition_to(self, new_state: RobotState):
        """状态转移"""
        print(f"状态转移: {self.current_state} -> {new_state}")
        self.current_state = new_state

    def should_process_blue_area(self, blue_detected: bool) -> bool:
        """判断是否应该处理蓝色区域"""
        if self.current_state == RobotState.NORMAL_NAVIGATION and blue_detected:
            return True
        elif self.current_state == RobotState.PERFORMING_ACTION:
            # 检查动作是否完成
            if self.action_start_time and (time.time() - self.action_start_time) >= self.action_duration:
                self.transition_to(RobotState.RETURN_TO_NORMAL)
            return False
        elif self.current_state == RobotState.RETURN_TO_NORMAL:
            # 短暂延迟后返回正常模式
            time.sleep(0.5)
            self.transition_to(RobotState.NORMAL_NAVIGATION)
            return False
        return False


async def perform_special_action(robot: Lite.Ysc_Lite3_Robot, action_name: str = "default"):
    """执行特殊动作"""
    print(f"开始执行特殊动作: {action_name}")

    # 这里可以根据不同的action_name执行不同的动作序列
    if action_name == "turn_around":
        # 示例：转身动作
        await robot.turn_right(0.3)
        await asyncio.sleep(0.1)
        await robot.turn_right(0.3)
        await asyncio.sleep(0.1)
    elif action_name == "look_around":
        # 示例：环顾四周
        await robot.turn_left(0.2)
        await asyncio.sleep(0.5)
        await robot.turn_right(0.4)
        await asyncio.sleep(0.5)
        await robot.turn_left(0.2)
    else:
        # 默认动作：停止并等待
        await robot.stop()
        await asyncio.sleep(2.0)

    print(f"特殊动作执行完成: {action_name}")


async def Deep():
    processor = DeepCamera.AsyncCameraProcessor()
    robot = Lite.Ysc_Lite3_Robot()

    # 初始化检测器
    RoadRobot = RoadFind.AsyncRoadDetector(roi_height=120, type="Yellow")
    BlueRobot = RoadFind.AsyncRoadDetector(roi_height=240, type="Blue")

    # 初始化状态机
    state_machine = StateMachine()

    # 历史数据缓冲区
    DegHistoryData = []
    StartTime = time.time()

    # 控制参数
    blue_action_executed = False  # 蓝色区域动作是否已执行
    normal_nav_paused = False  # 正常导航是否暂停

    # ROI定义
    rois = {
        "Left": (0, 215, 100, 50),
        "Right": (540, 215, 100, 50),
        "Mid": (270, 190, 100, 100),
        "MidDown": (270, 430, 100, 50),
        "MidUp": (270, 0, 100, 50),
    }

    try:
        async for distances in processor.measure_distances_stream(rois, interval=0.1):
            # 获取图像帧
            _, ColorFrams = await processor.get_frame_data()

            # 并行检测黄色道路和蓝色区域
            road_task = asyncio.create_task(RoadRobot.detect_road_with_visualization(ColorFrams))
            blue_task = asyncio.create_task(BlueRobot.detect_road_with_visualization(ColorFrams))

            RoadData, Fram = await road_task
            BlueRobotData, BlueFram = await blue_task

            # 获取中间区域最小距离
            MidMin = distances["Mid"]['distances'].get('min', 0)

            # 状态机决策
            current_state = state_machine.current_state

            if current_state == RobotState.NORMAL_NAVIGATION:
                # 正常导航模式
                if MidMin >= 0.5 and RoadData['road_detected'] and len(DegHistoryData) > 10:
                    # 保存角度历史数据
                    DegHistoryData.append(RoadData['angle'] + 90)
                    if len(DegHistoryData) > 20:  # 限制历史数据长度
                        DegHistoryData.pop(0)

                    # 平滑处理角度数据
                    smoothed_angle = round(sum(DegHistoryData[-6:-1]) / 5, 3) if len(DegHistoryData) >= 6 else RoadData[
                        'angle']

                    # 计算控制量
                    TurnMin = min(0.1, abs(smoothed_angle) / 1000)
                    MoveMin = min(0.1, abs(RoadData['position_offset']) / 1000)

                    # 执行导航控制
                    if smoothed_angle > 10:
                        print("turn_right")
                        await robot.turn_right(TurnMin)
                        await asyncio.sleep(TurnMin / 4)
                    elif smoothed_angle < -10:
                        print("turn_left")
                        await robot.turn_left(TurnMin)
                        await asyncio.sleep(TurnMin / 4)

                    if RoadData['position_offset'] < -10:
                        print("move_left")
                        await robot.move_left(MoveMin)
                        await asyncio.sleep(MoveMin / 4)
                    elif RoadData['position_offset'] > 10:
                        print("move_right")
                        await robot.move_right(MoveMin)
                        await asyncio.sleep(MoveMin / 4)

                    # 前进
                    await robot.move_forward(0.02)

                    # 检查是否检测到蓝色区域
                    if BlueRobotData['road_detected'] and not blue_action_executed:
                        print("检测到蓝色区域，准备执行特殊动作")
                        state_machine.transition_to(RobotState.BLUE_AREA_DETECTED)

                elif MidMin <= 0.5:
                    # 前方有障碍物
                    await robot.stop()
                    print(f"<警告> 前方距离过近: {MidMin}m")
                elif not RoadData['road_detected']:
                    # 未检测到道路
                    await robot.stop()
                    await robot.turn_left(0.01)

            elif current_state == RobotState.BLUE_AREA_DETECTED:
                # 检测到蓝色区域，导航到蓝色区域中心
                print("正在导航到蓝色区域...")

                # 停止前进
                await robot.stop()

                # 根据蓝色区域位置调整
                if BlueRobotData['angle'] > 10:
                    await robot.turn_right(0.05)
                    await asyncio.sleep(0.1)
                elif BlueRobotData['angle'] < -10:
                    await robot.turn_left(0.05)
                    await asyncio.sleep(0.1)

                if BlueRobotData['position_offset'] < -20:
                    await robot.move_left(0.05)
                    await asyncio.sleep(0.1)
                elif BlueRobotData['position_offset'] > 20:
                    await robot.move_right(0.05)
                    await asyncio.sleep(0.1)

                # 当位置对准时，前进到蓝色区域
                if abs(BlueRobotData['angle']) < 10 and abs(BlueRobotData['position_offset']) < 20:
                    print("已对准蓝色区域，准备执行动作")
                    await robot.move_forward(0.05)
                    await asyncio.sleep(0.5)
                    await robot.stop()

                    # 转移到执行动作状态
                    state_machine.transition_to(RobotState.PERFORMING_ACTION)
                    state_machine.action_start_time = time.time()

            elif current_state == RobotState.PERFORMING_ACTION:
                # 执行特殊动作
                print("执行特殊动作中...")

                # 停止所有运动
                await robot.stop()

                # 执行预定动作（这里可以扩展为不同的动作）
                if not blue_action_executed:
                    await perform_special_action(robot, "look_around")
                    blue_action_executed = True

                # 检查动作是否完成
                if state_machine.action_start_time and \
                        (time.time() - state_machine.action_start_time) >= state_machine.action_duration:
                    print("特殊动作执行完成，返回正常模式")
                    state_machine.transition_to(RobotState.RETURN_TO_NORMAL)
                    blue_action_executed = False  # 重置标志

            elif current_state == RobotState.RETURN_TO_NORMAL:
                # 返回正常模式前的短暂延迟
                print("返回正常导航模式...")
                await asyncio.sleep(1.0)
                state_machine.transition_to(RobotState.NORMAL_NAVIGATION)
                print("已返回正常导航模式")

            # 显示可视化结果
            try:
                ReturnFram = Fram * 0.4 + BlueFram * 0.6
                ReturnFram = ReturnFram.astype(np.uint8)

                # 在图像上显示当前状态
                state_text = f"状态: {state_machine.current_state.name}"
                cv2.putText(ReturnFram, state_text, (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("RoadData", ReturnFram)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"显示图像时出错: {e}")

    finally:
        # 清理资源
        await processor.close()
        await robot.close()
        cv2.destroyAllWindows()


async def main():
    """主函数"""
    print("启动机器狗控制系统...")
    print("状态说明:")
    print("1. NORMAL_NAVIGATION - 正常道路导航")
    print("2. BLUE_AREA_DETECTED - 检测到蓝色区域")
    print("3. PERFORMING_ACTION - 执行特殊动作")
    print("4. RETURN_TO_NORMAL - 返回正常模式")
    print("按 'q' 键退出程序\n")

    await Deep()


if __name__ == '__main__':
    asyncio.run(main())