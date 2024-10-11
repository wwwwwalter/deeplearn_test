import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('images/tuling.jpg')

# 检查图像是否成功读取
if image is None:
    raise ValueError("图像读取失败，请检查路径是否正确")

# 将BGR图像转换为RGB图像（因为OpenCV默认读取的是BGR格式）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 分离RGB通道
b, g, r = cv2.split(image_rgb)

# 对每个通道进行直方图均衡化
b_equalized = cv2.equalizeHist(b)
g_equalized = cv2.equalizeHist(g)
r_equalized = cv2.equalizeHist(r)

# 合并均衡化后的通道
equalized_image_rgb = cv2.merge((b_equalized, g_equalized, r_equalized))

# 显示原始图像和均衡化后的图像
plt.figure(figsize=(14, 8))

# 原始图像
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# 原始图像的直方图
plt.subplot(2, 2, 2)
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.title('Histogram of Original Image')

# 均衡化后的图像
plt.subplot(2, 2, 3)
plt.imshow(equalized_image_rgb)
plt.title('Equalized Image')
plt.axis('off')

# 均衡化后图像的直方图
plt.subplot(2, 2, 4)
for i, col in enumerate(colors):
    hist = cv2.calcHist([equalized_image_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.title('Histogram of Equalized Image')

plt.tight_layout()
plt.show()