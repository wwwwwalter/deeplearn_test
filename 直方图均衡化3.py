import cv2
import numpy as np
import matplotlib.pyplot as plt




def equalize_histogram(image):
    # 分离RGB通道
    b, g, r = cv2.split(image)

    # 对每个通道进行直方图均衡化
    b_equalized = cv2.equalizeHist(b)
    g_equalized = cv2.equalizeHist(g)
    r_equalized = cv2.equalizeHist(r)
    # 合并均衡化后的通道
    equalized_image = cv2.merge((b_equalized, g_equalized, r_equalized))
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
    equalized_frame_rgb = equalize_histogram(frame_rgb)

    # 显示原始图像和均衡化后的图像
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Equalized Frame', cv2.cvtColor(equalized_frame_rgb, cv2.COLOR_RGB2BGR))

    # 按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()