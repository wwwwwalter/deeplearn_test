import matplotlib.pyplot as plt
import numpy as np


# 定义生成数据的函数
def generate_data(num_points):
    points = []
    np.random.seed(42)  # 设置随机种子以便结果可重复
    # 循环生成指定数量的数据点
    for _ in range(num_points):
        x1 = np.random.uniform(-10., 10.)  # 随机生成x1值
        x2 = np.random.uniform(-10., 10.)  # 随机生成x2值
        y = 2 * x1 + 3 * x2 + np.random.normal()  # 根据线性关系计算y，并加入一点噪声
        points.append([x1, x2, y])  # 将数据点添加到列表中
    return np.array(points)  # 返回数据点数组


# 定义假设函数
def hypothesis(X, theta):
    return X @ theta  # 计算预测值


# 定义损失函数
def cost_function(X, y, theta):
    m = len(y)  # 数据点的数量
    predictions = hypothesis(X, theta)  # 计算预测值
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # 计算损失值（均方差）
    return cost  # 返回损失值


# # 定义梯度下降算法
# def gradient_descent(X, y, theta, learning_rate, num_iterations):
#     m = len(y)  # 数据点的数量
#     cost_history = [0] * num_iterations  # 初始化成本历史记录
#     # 循环进行梯度下降迭代
#     for iteration in range(num_iterations):
#         predictions = hypothesis(X, theta)  # 计算当前参数下的预测值
#         error = np.dot(X.transpose(), (predictions - y))  # 计算误差
#         theta -= (learning_rate / float(m)) * error  # 更新参数（标准梯度下降更新算法）
#         cost_history[iteration] = cost_function(X, y, theta)  # 记录当前迭代的成本(当前损失值/均方差)
#     return theta, cost_history  # 返回最终参数和成本历史

# 求偏导数向量(求梯度)
def compute_gradients(X, y, theta):
    m = len(y)  # 数据点的数量
    predictions = hypothesis(X, theta)  # 计算预测值
    gradients = (1 / m) * np.dot(X.T, (predictions - y))  # 计算偏导数
    return gradients


# 定义梯度下降算法
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)  # 数据点的数量
    cost_history = [0] * num_iterations  # 初始化成本历史记录

    # 循环进行梯度下降迭代
    for iteration in range(num_iterations):
        gradients = compute_gradients(X, y, theta)  # 计算偏导数向量（梯度）
        theta = theta - learning_rate * gradients  # 更新参数（标准梯度下降更新算法）
        cost_history[iteration] = cost_function(X, y, theta)  # 记录当前迭代的成本

    return theta, cost_history  # 返回最终参数和成本历史


# 生成数据
data = generate_data(100)  # 生成100个数据点

X = data[:, :2]  # 提取特征
y = data[:, 2]  # 提取目标变量

# 在特征矩阵X中添加一列全为1的向量，用于与θ0相乘
m = len(y)
X = np.hstack((np.ones((m, 1)), X))

# 初始化参数θ
theta = np.zeros(3)

# 设置学习率和迭代次数
learning_rate = 0.01
num_iterations = 1500

# 运行梯度下降算法
theta, cost_history = gradient_descent(X, y, theta, learning_rate, num_iterations)

# 输出最终的参数和最后一次迭代的成本
print('Final theta:', theta)
print('Cost history:', cost_history[-1])

# 绘制成本函数随迭代次数的变化图
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Iteration')  # 设置x轴标签
plt.ylabel('Cost')  # 设置y轴标签
plt.show()  # 显示图表

# 生成测试数据
test_data = generate_data(100)
test_X = test_data[:, :2]
test_y = test_data[:, 2]

# 在测试数据上添加一列全为1的向量
test_X = np.hstack((np.ones((len(test_X), 1)), test_X))

# 使用训练好的参数进行预测
predictions = hypothesis(test_X, theta)

# 绘制预测结果和真实值的对比图
plt.figure(figsize=(10, 6))
plt.scatter(test_X[:, 1], test_y, color='blue', label='True Values')
plt.scatter(test_X[:, 1], predictions, color='red', label='Predictions')
plt.xlabel('Feature X2')
plt.ylabel('Target Y')
plt.legend()
plt.title('Predictions vs True Values')
plt.show()

# # 绘制预测结果和真实值的对比图
# plt.figure(figsize=(10, 6))
# plt.scatter(test_X[:, 1], test_y, color='blue', label='True Values')
# plt.scatter(test_X[:, 1], predictions, color='red', label='Predictions')
#
# # 绘制拟合曲线
# sorted_indices = np.argsort(test_X[:, 1])
# sorted_x = test_X[sorted_indices, 1]
# sorted_predictions = predictions[sorted_indices]
#
# plt.plot(sorted_x, sorted_predictions, color='green', label='Fitted Curve')
#
# plt.xlabel('Feature X2')
# plt.ylabel('Target Y')
# plt.legend()
# plt.title('Predictions vs True Values with Fitted Curve')
# plt.show()

# 计算预测误差
mse = np.mean((predictions - test_y) ** 2)
print("Mean Squared Error:", mse)
