import matplotlib.pyplot as plt
import numpy as np


# 定义生成数据的函数
def generate_data(num_points, num_features=2):
    points = []
    np.random.seed(42)  # 设置随机种子以便结果可重复
    for _ in range(num_points):
        features = [np.random.uniform(-10., 10.) for _ in range(num_features)]
        y = sum([(i+2) * f for i, f in enumerate(features)]) + np.random.normal()
        points.append(features + [y])
    return np.array(points)


# 定义假设函数
def hypothesis(X, theta):
    return X @ theta


# 定义损失函数
def cost_function(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# 定义梯度下降算法
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = [0] * num_iterations
    for iteration in range(num_iterations):
        predictions = hypothesis(X, theta)
        error = np.dot(X.transpose(), (predictions - y))
        theta -= (learning_rate / float(m)) * error
        cost_history[iteration] = cost_function(X, y, theta)
    return theta, cost_history


# 生成数据
num_features = 2  # 特征数量
data = generate_data(100, num_features=num_features)  # 生成100个数据点
# print(data)

X = data[:, :num_features]  # 提取特征
y = data[:, num_features]  # 提取目标变量

# 在特征矩阵X中添加一列全为1的向量，用于与θ0相乘
m = len(y)
X = np.hstack((np.ones((m, 1)), X))


# 初始化参数θ
theta = np.zeros(num_features + 1)


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
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration')
plt.show()

# 生成测试数据
test_data = generate_data(100, num_features=num_features)
test_X = test_data[:, :num_features]
test_y = test_data[:, num_features]

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

# 计算预测误差
mse = np.mean((predictions - test_y) ** 2)
print("Mean Squared Error:", mse)
