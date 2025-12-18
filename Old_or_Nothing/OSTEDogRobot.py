import datetime
from DeepCamera import RealSenseUtils
import pyrealsense2 as rs
import numpy as np
import cv2
from LuPy_Lite3 import Ysc_Lite3_Robot
from Calibratingmidline import RobotController

def Create_RoiList(Img,Window,roi:dict):
    DoList = [SeeWhere[Key] for Key in roi]
    Draw_RoiList(Window,DoList)
    return {Key:RealSenseUtils.measure_distance_in_roi(Img, roi[Key]) for Key in roi}
def Draw_RoiList(Img,rio:list) ->None:
    for Where in rio:
        cv2.rectangle(Img,
                    (Where[0], Where[1]),
                    (Where[0] + Where[2], Where[1] + Where[3]),
                    (0, 255, 0), 2)
class RoiSelector:
    def __init__(self):
        self.StartPoint=None
        self.EndPoint=None
        self.Draw=None

    def  mouse_callback(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.StartPoint = (x,y)
            self.Draw = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.Draw:
                self.EndPoint = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.Draw = False
            self.EndPoint = (x,y)

            self.roi=(min(self.StartPoint[0],self.EndPoint[0]),min(self.StartPoint[1],self.EndPoint[1]),
                      abs(self.EndPoint[0]-self.StartPoint[0]),abs(self.EndPoint[1]-self.StartPoint[1])
                      )


if __name__ == '__main__':
    camera_info = RealSenseUtils.get_camera_info()

    for info in camera_info:
        print(f"摄像头: {info['name']}")
        print(f"  序列号: {info['serial']}")
        print(f"  固件版本: {info['firmware']}")

    #深度/彩图摄像头
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    Robot = RobotController()
    robot = Ysc_Lite3_Robot()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 创建深度叠加
            overlay = RealSenseUtils.create_depth_overlay(color_image, depth_image)

            height, width = depth_image.shape

            SeeWhere={
                "Left":(0,height//2-25,100,50),
                "Right":(width-100,height//2-25,100,50),
                "Mid":(width // 2 - 50, height // 2 - 50, 100, 100),
                "MidDown":(width//2-50,height-50,100,50),
            }

            # # 测量中心区域距离
            # roi=SeeWhere["Right"]
            distance_info = Create_RoiList(depth_image,overlay,SeeWhere)

            if distance_info:
                # 显示距离信息
                for Key1 in distance_info:
                    print(f"<UNK> {Key1} : {distance_info[Key1]}")
                    Data=distance_info[Key1]
                    Len = 0
                    XY = (SeeWhere[Key1][0], SeeWhere[Key1][1]+10)
                    if not distance_info[Key1]:
                        cv2.putText(overlay, f"None", XY, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 3)
                        break

                    for Key in Data:
                        XY = (SeeWhere[Key1][0], SeeWhere[Key1][1] + Len * 20+10)
                        cv2.putText(overlay, f"{Key}:{round(Data[Key], 2)}",XY , cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        Len+=1
                print(f"<TIME> : {datetime.datetime.now()}")
                #防撞机制
                try:
                    if not distance_info['Mid']['min']<=0.5:
                        robot.move_forward(0.1)
                    else:
                        Robot.send_data(0x21010130, 0, 0)
                except (TypeError, KeyError):
                    continue

            if cv2.EVENT_FLAG_LBUTTON:
                StartPoint=()

            cv2.imshow("Depth Overlay", overlay)
            cv2.imshow("Color", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()