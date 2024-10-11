import cv2
import numpy as np
import matplotlib.pyplot as plt


def block_equalize_histogram(image, block_size=(50, 50), overlap=(4, 4)):
    height, width, _ = image.shape
    block_height, block_width = block_size
    overlap_height, overlap_width = overlap

    # 计算每个块的实际大小
    effective_block_height = block_height - overlap_height
    effective_block_width = block_width - overlap_width

    # 初始化结果图像
    result = np.zeros_like(image, dtype=np.uint8)

    # 分割图像并处理每个块
    for y in range(0, height - block_height + 1, effective_block_height):
        for x in range(0, width - block_width + 1, effective_block_width):
            # 提取当前块
            block = image[y:y+block_height, x:x+block_width]

            # 分离RGB通道
            b, g, r = cv2.split(block)

            # 对每个通道进行直方图均衡化
            b_equalized = cv2.equalizeHist(b)
            g_equalized = cv2.equalizeHist(g)
            r_equalized = cv2.equalizeHist(r)

            # 合并均衡化后的通道
            equalized_block = cv2.merge((b_equalized, g_equalized, r_equalized))

            # 将均衡化后的块放入结果图像中
            result[y:y+block_height, x:x+block_width] = equalized_block

    return result


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

    # 进行分块直方图均衡化
    equalized_frame_rgb = block_equalize_histogram(frame_rgb)

    # 显示原始图像和均衡化后的图像
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Equalized Frame', cv2.cvtColor(equalized_frame_rgb, cv2.COLOR_RGB2BGR))

    # 按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
