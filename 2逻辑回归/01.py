"""
在训练的初始阶段，我们将要构建一个逻辑回归模型来预测，某个学生是否被大学录取。
设想你是大学相关部分的管理者，想通过申请学生两次测试的评分，来决定他们是否被录取。
现在你拥有之前申请学生的可以用于训练逻辑回归的训练样本集。对于每一个训练样本，
你有他们两次测试的评分和最后是被录取的结果。为了完成这个预测任务，
我们准备构建一个可以基于两次测试评分来评估录取可能性的分类模型。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入文件
path = '../data_sets/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
# print(data.head())

# 画出0和1的散点图
# isin([1,2])函数：查看数据中是否有1或2，返回逻辑数组
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax1.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
# 显示图例
ax1.legend()
# 增加坐标轴
ax1.set_xlabel('Exam 1 Score')
ax1.set_ylabel('Exam 2 Score')


# plt.show()


# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 检查是否可以运行
nums = np.arange(-10, 10, 1)
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(nums, sigmoid(nums), 'purple')


# plt.show()


# 代价函数
def cost(theta, X, y):
    # 将数据转为numpy matrix类型，高效运算
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    print(X.shape)
    print(y.shape)
    print(theta.shape)
    """
    (100, 3)
    (100, 1)
    (1, 3)
    """
    # multiply是普通乘法，不是矩阵乘法
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


# 初始设置
# 为了方便矩阵运算
data.insert(0, 'Ones', 1)
# print(data.head())
# 列
cols = data.shape[1]
# iloc的0：X中不包括X，只能到X-1.
# 获取除了最后一列的所有数据
X = data.iloc[:, 0:cols - 1]
# 获取最后一列所有数据
y = data.iloc[:, [cols - 1]]

theta = np.zeros(3)
print(theta)
# 计算初始参数的损失值
m_cost = cost(theta, X, y)
print(m_cost)


# 批量梯度下降
# 定义计算步长的函数:即返回代价函数对参数的求导的结果
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    # print("grad:", grad.shape)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
    return grad


grad = gradient(theta, X, y)
print(grad)


# 使用SciPy's truncated newton（TNC）实现寻找最优参数。
import scipy.optimize as opt
"""
opt.fmin_tnc（）函数 ：用于最优化（得到最小值）
基本参数：
func：优化的目标函数
x0：初值
fprime：提供优化函数func的梯度函数，不然优化函数func必须返回函数值和梯度，或者设置approx_grad=True
approx_grad :如果设置为True，会给出近似梯度
args：元组，是传递给优化函数的参数
"""
# 由于希望cost最小，因此目标函数时cost
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print(result)

m_cost1 = cost(result[0], X, y)
print(m_cost1)


# 预测函数
def predict(theta, X):
    X = np.matrix(X)
    theta = np.matrix(theta)
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
print("result:", predictions)
# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
# 计算预测值和样本真值是否一致
Y = np.matrix(y)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y)]
print("correct：", correct)
# map（函数名，数据）函数
# map将函数运用在每一个数据，返回一个列表
# 这里是将correct中所有数据转为int（都调用int函数）
accuracy = (sum(map(int, correct)) % len(correct))
print(f'accuracy = {accuracy}%')

