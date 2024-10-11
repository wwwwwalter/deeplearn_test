import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_histogram_hsv(image):
    # 将RGB图像转换为HSV图像
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 分离HSV通道
    h, s, v = cv2.split(hsv_image)

    # 对V通道进行直方图均衡化
    v_equalized = cv2.equalizeHist(v)

    # 合并均衡化后的V通道和其他通道
    equalized_hsv_image = cv2.merge((h, s, v_equalized))

    # 将HSV图像转换回RGB图像
    equalized_image = cv2.cvtColor(equalized_hsv_image, cv2.COLOR_HSV2RGB)
    return equalized_image

def plot_histogram(image, title='Histogram'):
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
    plt.title(title)
    plt.xlim([0, 256])
    plt.show()

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("无法打开摄像头")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将BGR图像转换为RGB图像
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行直方图均衡化
    equalized_frame_rgb = equalize_histogram_hsv(frame_rgb)

    # 显示原始图像和均衡化后的图像
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Equalized Frame', cv2.cvtColor(equalized_frame_rgb, cv2.COLOR_RGB2BGR))

    # 按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()