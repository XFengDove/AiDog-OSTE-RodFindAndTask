import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseUtils:
    def __init__(self):
        pass
    @staticmethod
    def get_camera_info():
        """获取摄像头信息"""
        ctx = rs.context()
        devices = ctx.query_devices()

        info = []
        for dev in devices:
            device_info = {
                'name': dev.get_info(rs.camera_info.name),
                'serial': dev.get_info(rs.camera_info.serial_number),
                'firmware': dev.get_info(rs.camera_info.firmware_version)
            }
            info.append(device_info)
        return info

    @staticmethod
    def save_depth_data(depth_image, filename):
        """保存深度数据为.npy文件"""
        np.save(filename, depth_image)
        print(f"深度数据已保存到 {filename}.npy")

    @staticmethod
    def create_depth_overlay(color_image, depth_image, min_dist=0.2, max_dist=3.0):
        """创建深度叠加可视化"""

        depth_normalized = np.clip(depth_image, min_dist * 1000, max_dist * 1000)
        depth_normalized = (depth_normalized - min_dist * 1000) / (max_dist * 1000 - min_dist * 1000)

        heatmap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 叠加到原始图像
        overlay = cv2.addWeighted(color_image, 0.7, heatmap, 0.3, 0)
        return overlay

    @staticmethod
    def measure_distance_in_roi(depth_image, roi)->dict[str:int]:
        """测量感兴趣区域的平均距离"""
        x, y, w, h = roi
        roi_depth = depth_image[y:y + h, x:x + w]

        # 移除零值（无效测量）
        roi_depth = roi_depth[roi_depth > 0]

        if len(roi_depth) == 0:
            return None

        avg_distance = np.mean(roi_depth) / 1000.0  # 转换为米
        min_distance = np.min(roi_depth) / 1000.0
        max_distance = np.max(roi_depth) / 1000.0

        return {
            'avg': avg_distance,
            'min': min_distance,
            'max': max_distance,
            'points': len(roi_depth)
        }


if __name__ == "__main__":
    # 1. 获取摄像头信息
    camera_info = RealSenseUtils.get_camera_info()
    for info in camera_info:
        print(f"摄像头: {info['name']}")
        print(f"  序列号: {info['serial']}")
        print(f"  固件版本: {info['firmware']}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    pipeline.start(config)
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

            # 测量中心区域距离
            height, width = depth_image.shape
            roi = (width // 2 - 50, height // 2 - 50, 100, 100)
            distance_info = RealSenseUtils.measure_distance_in_roi(depth_image, roi)

            if distance_info:
                # 绘制ROI矩形
                cv2.rectangle(overlay,
                              (roi[0], roi[1]),
                              (roi[0] + roi[2], roi[1] + roi[3]),
                              (0, 255, 0), 2)

                # 显示距离信息
                AvgText = f"Avg: {distance_info['avg']:.2f}m"
                MinText = f"Min: {distance_info['min']:.2f}m"
                cv2.putText(overlay, AvgText, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(overlay, MinText, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Depth Overlay", overlay)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()