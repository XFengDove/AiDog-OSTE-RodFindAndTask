import time
from numba import njit,prange
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import numba_cuda as cuda
# 读取图像
IMG = cv2.imread('img/Road_1.jpg')[::3,::4]
print(IMG.shape)


@njit(nogil=True,parallel=True)
def FindDATA(Fram:np.ndarray):
    if Fram.shape >=3:
        raise "请使用二值图或者二维图像"
    frame_data = Fram.copy()
    height, width = Fram.shape
    #检索像素位置
    results = []
    for y in prange(height):
        for x in prange(width):
            if frame_data[y][x] == 0:
                results.append((y,x))
    return results
@njit(nogil=True,parallel=True)
def RessBy(Fram:np.ndarray,r:int=20):
    # 划分检索区域
    Data = FindDATA(Fram)

    FramR={}
    for y,x in Data:
        Y1,X1=y-r,x+r
        Y2,X2=y+r,x+r

        height, width = Y2-Y1,X2-X1

        Roi=(Y1,X1,height,width)

        RoiPiData=[]
        for Roi_Y in prange(Y1,Y2):
            for Roi_X in prange(X1,X2):
                Roi_r=np.sqrt((Roi_X-Y1)**2+(Roi_Y-Y2)**2)
                if Roi_r<r:
                    RoiPiData.append((Roi_Y,Roi_X))
        FramR[f"({y},{x})"]=RoiPiData
    return FramR

@njit(nogil=True,parallel=True)
def getEight(YX:tuple):
    Y,X=YX
    results=[]
    for y in prange(Y-1,Y+1):
        for x in prange(X-1,X+1):
            if (y,x) == YX:
                continue
            results.append((y,x))
    return results

#检索区域内是否有非连接像素
@njit(nogil=True,parallel=True)
def HelloEverybody(Fram:np.ndarray):
    DATA= RessBy(Fram)

    for Key in DATA:
        DataR=DATA[Key]
        DataO=tuple(Key.copy())

        # 检索连接像素
        # 创建结果池
        functions=[]
        functions.append(DataO)#加入首要结果
        for ii in functions:
            for i in getEight(ii):
                Y,X=i
                if Fram[Y,X]==0 and (Y,X) not in functions and (Y,X) in DataR:
                    functions.append((Y,X))
                else:
                    continue

        #清零资源
        DataR=DataR - functions
        for Y,X in DataR:
            if Fram[Y,X]==0:
                #合并区域 算法区域
                pass






#连接最短路径


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
    """
    将图片转化为二值图

    :param func:
    :param Fram:
    :param core:
    :return:
    """
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

#常规调用
# ST=time.time()
# IMG_Goto_Grayscale_Image(np.array(IMG[:10,:10]))
# print(time.time()-ST)


times=[]
#多核调用
ST=time.time()
img=main(IMG_Goto_Grayscale_Image,IMG,1)
times.append(time.time()-ST)

ST=time.time()
img=main(IMG_Goto_Grayscale_Image,IMG,6)
times.append(time.time()-ST)
print(max(times),min(times),sum(times)/len(times))
print(times)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

