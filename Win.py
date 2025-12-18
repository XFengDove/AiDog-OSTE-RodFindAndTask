import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
from numba import prange,njit
import numpy as np
import src.OSTEDogRobot_V1_3 as DeepCamera
import src.RoadFind_V1_3 as RoadFind
import src.Ysc_Lite_V1_2 as Lite
import time
import copy

rois = {
    "Left": (0, 215, 100, 50),
    "Right": (540, 215, 100, 50),
    "Mid": (270, 190, 100, 100),
    "MidDown": (270, 430, 100, 50),
    "MidUp": (270, 0, 100, 50),
}

State = {
    "导航": False,
    "寻路": True,
    "特殊动作": False,
    "动作": {
        "nst": False,
        "tkb": False,
        "dzh": False,
    }
}

@njit(nogil=True,parallel=True)
def DEEPCOLOR(fram:np.ndarray)->np.ndarray:
    fram=fram[:,:,0]
    results =np.zeros_like(fram)
    height, width = fram.shape[:2]
    for y in prange(height):
        for x in prange(width):
            if fram[y,x] >75:
                results[y,x] = 255
            else:
                results[y,x] = 0
    return results

@njit(nogil=True,parallel=True,cache=True)
def IMG_Goto_Grayscale_Image(img:np.ndarray)->np.ndarray:
    """

    :param img:
    :return:
    """
    results=np.zeros_like(img,dtype=np.uint8)

    height,width=img.shape[:2]
    for y in prange(height):
        for x in prange(width):
            Data=img[y, x, 0] * 0.1 + img[y,x, 1] * 0.1 + img[y, x, 2] * 0.8
            results[y, x] = [int(Data),int(Data),int(Data)]
    return DEEPCOLOR(results)
def main(func,Fram:np.ndarray,core:int)->np.ndarray:
    with ThreadPoolExecutor(core) as executor:
        height,width=Fram.shape[:2]
        step=height//core
        futures=[]
        for i in range(0,len(Fram),step):
            futures.append(executor.submit(func,Fram[i:i+step]))

        results=[]
        for future in futures:
            frame_data = future.result()
            results.append(frame_data)

    return np.vstack(results)


async def Deep():
    pass_ = 1
    processor = DeepCamera.AsyncCameraProcessor()
    robot = Lite.Ysc_Lite3_Robot()

    RoadRobot = RoadFind.AsyncRoadDetector(roi_height=220, type="Yellow")

    BlueRobot = RoadFind.AsyncRoadDetector(roi_height=250, type="Blue")

    DegHistoryData = []
    StartTime = time.time()
    ActionHistory = []
    Action = 1
    Blue = False
    Road = False
    try:
        # 持续测量并显示
        State["寻路"] = True
        async for distances in processor.measure_distances_stream(rois, interval=0.05):

            print(f"时间: {distances['Left']['timestamp'].strftime('%H:%M:%S.%f')}")
            MidMin = distances["Mid"]['distances'].get('min', 0)

            DeepFram, ColorFrams = await processor.get_frame_data()
            RoadData, Fram = await RoadRobot.detect_road_with_visualization(ColorFrams)
            BlueRobotData, BlueFram = await BlueRobot.detect_road_with_visualization(ColorFrams)

            # print(BlueFram.shape)
            BlueFram = main(IMG_Goto_Grayscale_Image,BlueFram,6)
            # for Data in RoadData:
            #     print(f"<RoadData - >> {Data} : {RoadData[Data]}")

            DegHistoryData.append(RoadData['angle'] + 90)

            NowTime = time.time()
            if MidMin >= 0.5 and (RoadData['road_detected'] or BlueRobotData['road_detected']) and len(
                    DegHistoryData) > 10:

                # RoadData['angle'] = round(sum(DegHistoryData[-6:-1]) / 5, 3)

                print(f"{RoadData['angle']}    |   {RoadRobot.roi_height}")

                TurnMin = min(0.1, abs(RoadData['angle']) / 1000)
                MoveMin = min(0.1, abs(RoadData['position_offset']) / 1000)
                print(TurnMin, MoveMin)

                if BlueRobotData['road_detected']:
                    State["寻路"] = False
                    State["导航"] = True
                    State["特殊动作"] = False
                if State["寻路"]:
                    if RoadData['angle'] > 20 and NowTime - StartTime > 1:
                        print("turn_right")
                        await robot.turn_right(TurnMin)
                        await asyncio.sleep(TurnMin / 4)
                    elif RoadData['angle'] < -20 and NowTime - StartTime > 1:
                        print("turn_left")
                        await robot.turn_left(0.1)
                        await asyncio.sleep(TurnMin / 4)
                    if RoadData['position_offset'] < -10:
                        print("move_left")
                        await robot.move_left(MoveMin)
                        await asyncio.sleep(MoveMin / 4)
                    elif RoadData['position_offset'] > 10:
                        print("move_right")
                        await robot.move_right(MoveMin)
                        await asyncio.sleep(MoveMin / 4)
                    await robot.move_forward(0.1)

                    if BlueRobotData['road_detected']:
                        State["寻路"] = False
                        State["导航"] = True
                        State["特殊动作"] = False
                elif State["导航"]:
                    if BlueRobotData['angle'] > 20:
                        print("turn_right")
                        await robot.turn_right(TurnMin)
                        await asyncio.sleep(TurnMin / 2)
                        ActionHistory.append("turn_right")
                    elif BlueRobotData['angle'] < -20:
                        print("turn_left")
                        await robot.turn_left(0.1)
                        await asyncio.sleep(TurnMin / 2)
                        ActionHistory.append("turn_left")
                    if BlueRobotData['position_offset'] < -10:
                        print("move_left")
                        await robot.move_left(MoveMin)
                        await asyncio.sleep(MoveMin / 2)
                        ActionHistory.append("move_left")
                    elif BlueRobotData['position_offset'] > 10:
                        print("move_right")
                        await robot.move_right(MoveMin)
                        await asyncio.sleep(MoveMin / 2)
                        ActionHistory.append("move_right")
                    await robot.move_forward(0.02)

                    print(BlueRobotData['road_detected'])

                    if not BlueRobotData['road_detected']:
                        State["寻路"] = False
                        State["导航"] = False
                        State["特殊动作"] = True
                elif State["特殊动作"]:
                    print("特殊动作模式")
                    if BlueRobotData['road_detected']:
                        """
                        防止意外丢失目标
                        """
                        print("回归导航模式")
                        State["寻路"] = False
                        State["导航"] = True
                        State["特殊动作"] = False
                    else:
                        if pass_ == 1 or pass_ == 2 or pass_ == 3:
                            State["特殊动作"] = False
                            await robot.dz_dzh(3)
                            await asyncio.sleep(6)
                            pass_ += 1
                        State["寻路"] = True
                        State["导航"] = False
                        State["特殊动作"] = False

                print(State)
            elif MidMin <= 0.5:
                await robot.stop()
                print(f"<Warring>{MidMin}</Warring>")
            elif not RoadData['road_detected']:
                await robot.stop()
                if State["寻路"]:
                    await robot.turn_left(0.01)
                    await asyncio.sleep(0.01)
            # ReturnFram = BlueFram * 0.6 + Fram * 0.4
            # ReturnFram = ReturnFram.astype(np.uint8)
            try:
                cv2.imshow("BlueFram", BlueFram)
                cv2.imshow("RoadData", Fram)
            except:
                pass

    finally:
        await processor.close()
        await robot.close()
        cv2.destroyAllWindows()


asyncio.run(Deep())
