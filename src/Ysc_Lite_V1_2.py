import socket, struct, time, asyncio
from enum import IntEnum
from typing import Optional
import threading


class InstructionType(IntEnum):
    SIMPLE = 0
    COMPLEX = 1


class Ysc_Lite3_Robot:
    def __init__(self) -> None:
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = ("192.168.1.120", 43893)
        self._heartbeat_task = None
        self._heartbeat_running = False
        self._current_action_task = None  # 跟踪当前正在执行的动作任务

    @staticmethod
    def __map_speed(speed: Optional[float], default: int) -> int:
        if speed is None:
            return default
        return int(speed * 32767)

    @staticmethod
    def __map_yaw_rate(rate: Optional[float], default: int) -> int:
        if rate is None:
            return default
        return int(rate * 21845.3)

    async def send_data(self, code, value, type):
        """发送控制指令"""
        data = struct.pack("<3i", code, value, type)
        self.udp_socket.sendto(data, self.send_addr)
    async def stop(self):
        """停止所有运动并清理资源"""
        # 发送停止指令
        await self.send_data(0x21010130, 0, 0)  # 停止前进
        await self.send_data(0x21010131, 0, 0)  # 停止左右移动
        await self.send_data(0x21010135, 0, 0)  # 停止旋转
        await asyncio.sleep(0.1)  # 确保指令发送完成
    async def send_instruction(self, code: int, value: float, type_: InstructionType) -> None:
        """异步发送指令"""
        data = struct.pack("<3i", code, value, type_.value)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.udp_socket.sendto, data, self.send_addr)

    async def __check_heartbeat(self) -> None:
        """异步心跳检测"""
        self._heartbeat_running = True
        while self._heartbeat_running:
            await self.send_instruction(0x21040001, 0, InstructionType.SIMPLE)
            await asyncio.sleep(0.5)

    def start_heart_checking(self) -> None:
        """启动心跳检测"""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self.__check_heartbeat())

    def stop_heart_checking(self) -> None:
        """停止心跳检测"""
        self._heartbeat_running = False
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

    def cancel_current_action(self) -> None:
        """取消当前正在执行的动作"""
        if self._current_action_task and not self._current_action_task.done():
            self._current_action_task.cancel()
            self._current_action_task = None

    async def stand_up(self) -> None:
        """起立 - 异步版本"""
        await self.send_instruction(0x21010202, 0, InstructionType.SIMPLE)
        await self.move_mode()
        await asyncio.sleep(2)

    async def sit(self) -> None:
        """趴下 - 异步版本"""
        await self.send_instruction(0x21010202, 0, InstructionType.SIMPLE)
        await asyncio.sleep(2)

    async def move_forward(self, duration: float, speed: Optional[float] = None) -> None:
        """向前走 - 异步版本"""
        self.cancel_current_action()  # 取消之前的动作
        self._current_action_task = asyncio.create_task(self._move_forward_impl(duration, speed))

    async def _move_forward_impl(self, duration: float, speed: Optional[float] = None) -> None:
        start_time = time.time()
        while time.time() - start_time < duration:
            await self.send_instruction(
                0x21010130, self.__map_speed(speed, 13000), InstructionType.SIMPLE
            )
            await asyncio.sleep(0.05)

    async def move_backward(self, duration: float, speed: Optional[float] = None) -> None:
        """向后走 - 异步版本"""
        self.cancel_current_action()
        self._current_action_task = asyncio.create_task(self._move_backward_impl(duration, speed))

    async def _move_backward_impl(self, duration: float, speed: Optional[float] = None) -> None:
        start_time = time.time()
        while time.time() - start_time < duration:
            await self.send_instruction(
                0x21010131, -self.__map_speed(speed, 13000), InstructionType.SIMPLE
            )
            await asyncio.sleep(0.05)

    async def move_left(self, duration: float, speed: Optional[float] = None) -> None:
        """向左走 - 异步版本"""
        self.cancel_current_action()
        self._current_action_task = asyncio.create_task(self._move_left_impl(duration, speed))

    async def _move_left_impl(self, duration: float, speed: Optional[float] = None) -> None:
        start_time = time.time()
        while time.time() - start_time < duration:
            await self.send_instruction(
                0x21010131, -self.__map_speed(speed, 26000), InstructionType.SIMPLE
            )
            await asyncio.sleep(0.05)

    async def move_right(self, duration: float, speed: Optional[float] = None) -> None:
        """向右走 - 异步版本"""
        self.cancel_current_action()
        self._current_action_task = asyncio.create_task(self._move_right_impl(duration, speed))

    async def _move_right_impl(self, duration: float, speed: Optional[float] = None) -> None:
        start_time = time.time()
        while time.time() - start_time < duration:
            await self.send_instruction(
                0x21010131, self.__map_speed(speed, 26000), InstructionType.SIMPLE
            )
            await asyncio.sleep(0.05)

    async def turn_left(self, duration: float, rate: Optional[float] = None) -> None:
        """向左转 - 异步版本"""
        self.cancel_current_action()
        self._current_action_task = asyncio.create_task(self._turn_left_impl(duration, rate))

    async def _turn_left_impl(self, duration: float, rate: Optional[float] = None) -> None:
        start_time = time.time()
        while time.time() - start_time < duration:
            await self.send_instruction(
                0x21010135, -self.__map_yaw_rate(rate, 10000), InstructionType.SIMPLE
            )
            await asyncio.sleep(0.02)

    async def turn_right(self, duration: float, rate: Optional[float] = None) -> None:
        """向右转 - 异步版本"""
        self.cancel_current_action()
        self._current_action_task = asyncio.create_task(self._turn_right_impl(duration, rate))

    async def _turn_right_impl(self, duration: float, rate: Optional[float] = None) -> None:
        start_time = time.time()
        while time.time() - start_time < duration:
            await self.send_instruction(
                0x21010135, self.__map_yaw_rate(rate, 10000), InstructionType.SIMPLE
            )
            await asyncio.sleep(0.02)

    async def turn_left_90_degree(self) -> None:
        """向左转90度 - 异步版本"""
        await self.send_instruction(0x21010C0A, 13, InstructionType.SIMPLE)

    async def turn_right_90_degree(self) -> None:
        """向右转90度 - 异步版本"""
        await self.send_instruction(0x21010C0A, 14, InstructionType.SIMPLE)

    async def turn_around_180_degree(self) -> None:
        """向后转180度 - 异步版本"""
        await self.send_instruction(0x21010C0A, 15, InstructionType.SIMPLE)

    async def dz_dzh(self, duration: float) -> None:
        """打招呼 - 异步版本"""
        self.cancel_current_action()
        self._current_action_task = asyncio.create_task(self._dz_dzh_impl(duration))

    async def _dz_dzh_impl(self, duration: float) -> None:
        start_time = time.time()
        while time.time() - start_time < duration:
            await self.send_instruction(0x21010507, 0, InstructionType.SIMPLE)
            await asyncio.sleep(0.5)

    async def dz_nst(self, duration: float) -> None:
        """扭身体 - 异步版本"""
        self.cancel_current_action()
        self._current_action_task = asyncio.create_task(self._dz_nst_impl(duration))

    async def _dz_nst_impl(self, duration: float) -> None:
        start_time = time.time()
        while time.time() - start_time < duration:
            await self.send_instruction(0x21010204, 0, InstructionType.SIMPLE)
            await asyncio.sleep(0.5)

    async def dz_tkb(self, duration: float) -> None:
        """太空步 - 异步版本"""
        self.cancel_current_action()
        self._current_action_task = asyncio.create_task(self._dz_tkb_impl(duration))

    async def _dz_tkb_impl(self, duration: float) -> None:
        start_time = time.time()
        while time.time() - start_time < duration:
            await self.send_instruction(0x2101030C, 0, InstructionType.SIMPLE)
            await asyncio.sleep(0.5)

        if time.time() - start_time > 6:
            await self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)
            await asyncio.sleep(0.5)

    # ============ 模式切换（异步版本） ============

    async def stay_mode(self) -> None:
        """切换成原地模式 - 异步版本"""
        await self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)

    async def move_mode(self) -> None:
        """切换成移动模式 - 异步版本"""
        await self.send_instruction(0x21010D06, 0, InstructionType.SIMPLE)

    async def low_speed(self) -> None:
        """切换成平地低速步态 - 异步版本"""
        await self.send_instruction(0x21010300, 0, InstructionType.SIMPLE)

    async def medium_speed(self) -> None:
        """切换成平地中速步态 - 异步版本"""
        await self.send_instruction(0x21010307, 0, InstructionType.SIMPLE)

    async def high_speed(self) -> None:
        """切换成平地高速步态 - 异步版本"""
        await self.send_instruction(0x21010303, 0, InstructionType.SIMPLE)

    async def toggle_normal_creep(self) -> None:
        """正常/匍匐切换 - 异步版本"""
        await self.send_instruction(0x21010406, 0, InstructionType.SIMPLE)

    async def ground_grabbing_and_obstacle_crossing(self) -> None:
        """切换成抓地越障步态 - 异步版本"""
        await self.send_instruction(0x21010402, 0, InstructionType.SIMPLE)

    async def universal_obstacle_crossing(self) -> None:
        """切换成通用越障步态 - 异步版本"""
        await self.send_instruction(0x21010401, 0, InstructionType.SIMPLE)

    async def high_step_obstacle_crossing(self) -> None:
        """切换成高踏步越障步态 - 异步版本"""
        await self.send_instruction(0x21010301, 0, InstructionType.SIMPLE)

    async def enable_continuous_movement(self) -> None:
        """开启持续运动 - 异步版本"""
        await self.send_instruction(0x21010C06, -1, InstructionType.SIMPLE)

    async def disable_continuous_movement(self) -> None:
        """关闭持续运动 - 异步版本"""
        await self.send_instruction(0x21010C06, 2, InstructionType.SIMPLE)

    async def automatic_mode(self) -> None:
        """切换成自主模式 - 异步版本"""
        await self.send_instruction(0x21010C03, 0, InstructionType.SIMPLE)

    async def forward_and_backward_movement_speed_(self, speed: float) -> None:
        """自主模式下指定前后平移速度 - 异步版本"""
        await self.send_instruction(0x0140, speed, InstructionType.COMPLEX)

    async def left_and_right_movement_speed_(self, speed: float) -> None:
        """自主模式下指定左右平移速度 - 异步版本"""
        await self.send_instruction(0x0145, speed, InstructionType.COMPLEX)

    async def rotation_angle_speed(self, speed: float) -> None:
        """自主模式下指定旋转角速度 - 异步版本"""
        await self.send_instruction(0x0141, 0.5, InstructionType.COMPLEX)

    async def manual_mode(self) -> None:
        """切换成手动模式 - 异步版本"""
        await self.send_instruction(0x21010C02, 0, InstructionType.SIMPLE)

    async def close(self) -> None:
        """关闭机器人连接"""
        self.stop_heart_checking()
        self.cancel_current_action()
        if self._heartbeat_task and not self._heartbeat_task.done():
            await self._heartbeat_task
        self.udp_socket.close()


# 同步版本包装器（可选，用于向后兼容）
class SyncRobotWrapper:
    """同步包装器，用于在不支持异步的环境中使用"""

    def __init__(self):
        self.robot = Ysc_Lite3_Robot()
        self.loop = asyncio.new_event_loop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loop.run_until_complete(self.robot.close())
        self.loop.close()

    def stand_up(self):
        self.loop.run_until_complete(self.robot.stand_up())

    def sit(self):
        self.loop.run_until_complete(self.robot.sit())

    def move_forward(self, duration: float, speed: Optional[float] = None):
        self.loop.run_until_complete(self.robot.move_forward(duration, speed))

    # ... 其他方法的同步包装


if __name__ == "__main__":
    # 测试代码示例
    async def test_robot():
        robot = Ysc_Lite3_Robot()

        try:
            # 启动心跳
            robot.start_heart_checking()

            # 执行动作
            await robot.stand_up()
            await robot.move_forward(2.0, 0.5)
            await robot.turn_left(1.0, 1.0)
            await robot.sit()

        finally:
            await robot.close()


    asyncio.run(test_robot())