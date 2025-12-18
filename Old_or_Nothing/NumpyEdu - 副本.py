import time
from numba import njit,prange
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

# 读取图像
IMG = cv2.imread('img/Road_1.jpg')


@njit(nogil=True)
def IMG_Goto_Grayscale_Image(img:np.ndarray)->np.ndarray:
    results=np.zeros_like(img,dtype=np.uint8)
    height,width=img.shape[:2]
    for y in range(height):
        for x in range(width):
            Data=img[y, x, 0] * 0.299 + img[y,x, 1] * 0.587 + img[y, x, 2] * 0.114
            results[y, x] = [int(Data),int(Data),int(Data)]
    return results
def main(func,Fram:np.ndarray,core:int)->np.ndarray:
    with ThreadPoolExecutor(6) as executor:
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
ST=time.time()
IMG_Goto_Grayscale_Image(np.array(IMG))
print(time.time()-ST)

#多核调用
ST=time.time()
img=main(IMG_Goto_Grayscale_Image,IMG,4)
print(time.time()-ST)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

