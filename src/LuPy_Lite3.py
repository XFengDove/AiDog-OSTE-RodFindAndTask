import socket, struct, time, threading
from enum import IntEnum
from typing import Optional


class InstructionType(IntEnum):
    SIMPLE = 0
    COMPLEX = 1


class Ysc_Lite3_Robot:
    def __init__(self) -> None:
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = ("192.168.1.120", 43893)
        self.heart_checking_task = threading.Thread(target=self.__check_heartbeat)

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

    def send_instruction(self, code: int, value: float, type_: InstructionType) -> None:
        data = struct.pack("<3i", code, value, type_.value)
        self.udp_socket.sendto(data, self.send_addr)

    def __check_heartbeat(self) -> None:
        while True:
            self.send_instruction(0x21040001, 0, InstructionType.SIMPLE)
            time.sleep(0.5)

    def start_heart_checking(self) -> None:
        self.heart_checking_task.start()

    # 动作
    def stand_up(self) -> None:
        """起立"""
        self.send_instruction(0x21010202, 0, InstructionType.SIMPLE)
        self.move_mode()
        time.sleep(2)

    def sit(self) -> None:
        """趴下"""
        self.send_instruction(0x21010202, 0, InstructionType.SIMPLE)
        time.sleep(2)

    def move_forward(self, duration: float, speed: Optional[float] = None) -> None:
        """向前走，speed参数单位为m/s，范围为[0.2, 1]"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(
                0x21010130, self.__map_speed(speed, 13000), InstructionType.SIMPLE
            )
            time.sleep(0.05)

    def move_backward(self, duration: float, speed: Optional[float] = None) -> None:
        """向后走，speed参数单位为m/s，范围为[0.2, 1]"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(
                0x21010131, -self.__map_speed(speed, 13000), InstructionType.SIMPLE
            )
            time.sleep(0.05)

    def move_left(self, duration: float, speed: Optional[float] = None) -> None:
        """向左走，speed参数单位为m/s，范围为[0.4, 1]"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(
                0x21010131, -self.__map_speed(speed, 26000), InstructionType.SIMPLE
            )
            time.sleep(0.05)

    def move_right(self, duration: float, speed: Optional[float] = None) -> None:
        """向右走，speed参数单位为m/s，范围为[0.4, 1]"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(
                0x21010131, self.__map_speed(speed, 26000), InstructionType.SIMPLE
            )
            time.sleep(0.05)

    def turn_left(self, duration, rate: Optional[float] = None) -> None:
        """向左转，rate参数单位为rad/s，范围为[0.5, 1.5]"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(
                0x21010135, -self.__map_yaw_rate(rate, 10000), InstructionType.SIMPLE
            )
            time.sleep(0.02)

    def turn_right(self, duration, rate: Optional[float] = None) -> None:
        """向右转，rate参数单位为rad/s，范围为[0.5, 1.5]"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(
                0x21010135, self.__map_yaw_rate(rate, 10000), InstructionType.SIMPLE
            )
            time.sleep(0.02)

    def turn_left_90_degree(self) -> None:
        """向左转90度"""
        self.send_instruction(0x21010C0A, 13, InstructionType.SIMPLE)

    def turn_right_90_degree(self) -> None:
        """向右转90度"""
        self.send_instruction(0x21010C0A, 14, InstructionType.SIMPLE)

    def turn_around_180_degree(self) -> None:
        """向后转180度"""
        self.send_instruction(0x21010C0A, 15, InstructionType.SIMPLE)

    def dz_dzh(self, duration) -> None:
        """打招呼"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(0x21010507, 0, InstructionType.SIMPLE)
            time.sleep(0.5)

    def dz_nst(self, duration) -> None:
        """扭身体"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(0x21010204, 0, InstructionType.SIMPLE)
            time.sleep(0.5)

    def dz_tkb(self, duration) -> None:
        """太空步"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(0x2101030C, 0, InstructionType.SIMPLE)
            time.sleep(0.5)
        while time.time() - start_time > 6:
            self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)
            time.sleep(0.5)

    def dz_1(self, duration) -> None:
        """自定动作2 原地模式右倾倒"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)
            self.send_instruction(0x21010131, 30000, InstructionType.SIMPLE)
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)
            time.sleep(0.1)

    def dz_2(self, duration) -> None:
        """自定动作2 原地模式左倾倒"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)
            self.send_instruction(0x21010131, -30000, InstructionType.SIMPLE)
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)
            time.sleep(0.1)

    def dz_3(self, duration) -> None:
        """自定动作3 原地模式后倾倒"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)
            self.send_instruction(0x21010130, -30000, InstructionType.SIMPLE)
        start_time = time.time()
        while time.time() - start_time < duration:  # > 不回正
            self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)
            time.sleep(0.1)

    def dz_4(self, duration) -> None:
        """自定动作4 原地模式前倾倒"""
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)
            self.send_instruction(0x21010130, 30000, InstructionType.SIMPLE)
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)
            time.sleep(0.1)

    # 模式
    def stay_mode(self) -> None:
        """切换成原地模式"""
        self.send_instruction(0x21010D05, 0, InstructionType.SIMPLE)

    def move_mode(self) -> None:
        """切换成移动模式"""
        self.send_instruction(0x21010D06, 0, InstructionType.SIMPLE)

    def low_speed(self) -> None:
        """切换成平地低速步态"""
        self.send_instruction(0x21010300, 0, InstructionType.SIMPLE)

    def medium_speed(self) -> None:
        """切换成平地中速步态"""
        self.send_instruction(0x21010307, 0, InstructionType.SIMPLE)

    def high_speed(self) -> None:
        """切换成平地高速步态"""
        self.send_instruction(0x21010303, 0, InstructionType.SIMPLE)

    def toggle_normal_creep(self) -> None:
        """正常/匍匐切换"""
        self.send_instruction(0x21010406, 0, InstructionType.SIMPLE)

    def ground_grabbing_and_obstacle_crossing(self) -> None:
        """切换成抓地越障步态"""
        self.send_instruction(0x21010402, 0, InstructionType.SIMPLE)

    def universal_obstacle_crossing(self) -> None:
        """切换成通用越障步态"""
        self.send_instruction(0x21010401, 0, InstructionType.SIMPLE)

    def high_step_obstacle_crossing(self) -> None:
        """切换成高踏步越障步态"""
        self.send_instruction(0x21010301, 0, InstructionType.SIMPLE)

    def enable_continuous_movement(self) -> None:
        """开启持续运动"""
        self.send_instruction(0x21010C06, -1, InstructionType.SIMPLE)

    def disable_continuous_movement(self) -> None:
        """关闭持续运动"""
        self.send_instruction(0x21010C06, 2, InstructionType.SIMPLE)

    def automatic_mode(self) -> None:
        """切换成自主模式"""
        self.send_instruction(0x21010C03, 0, InstructionType.SIMPLE)

    def forward_and_backward_movement_speed_(self, speed: float) -> None:
        """自主模式下指定前后平移速度 speed参数单位为m/s，范围为[-1.0,1.0]"""
        self.send_instruction(0x0140, speed, InstructionType.COMPLEX)

    def left_and_right_movement_speed_(self, speed: float) -> None:
        """自主模式下指定左右平移速度 speed参数单位为m/s，范围为[-0.5,0.5]"""
        self.send_instruction(0x0145, speed, InstructionType.COMPLEX)

    def rotation_angle_speed(self, speed: float) -> None:
        """自主模式下指定旋转角速度 speed参数单位为rad/s，范围为[-1.5,1.5]"""
        self.send_instruction(0x0141, 0.5, InstructionType.COMPLEX)

    def manual_mode(self) -> None:
        """切换成手动模式"""
        self.send_instruction(0x21010C02, 0, InstructionType.SIMPLE)


if __name__ == "__main__":
    robot = Ysc_Lite3_Robot()
    # 测试代码还没写

