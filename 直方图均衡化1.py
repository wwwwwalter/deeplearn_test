import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('images/tuling.jpg', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if image is None:
    raise ValueError("图像读取失败，请检查路径是否正确")

# 直方图均衡化
equalized_image = cv2.equalizeHist(image)

# 显示原始图像和均衡化后的图像
plt.figure(figsize=(12, 6))

# 原始图像
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
# plt.axis('off')

# 原始图像的直方图
plt.subplot(2, 2, 2)
plt.hist(image.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.title('Histogram of Original Image')

# 均衡化后的图像
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# 均衡化后图像的直方图
plt.subplot(2, 2, 4)
plt.hist(equalized_image.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
plt.title('Histogram of Equalized Image')

plt.tight_layout()
plt.show()