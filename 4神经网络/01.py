import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

# 导入数据
data = loadmat("../data_sets/ex4data1.mat")
print(data)

X = data['X']
y = data['y']

# print(X.shape)
# # print(y.shape)

# 对y进行one-hot编码
# one-hot 编码将类标签n（k类）转换为长度为k的向量，其中索引n为“hot”（1），而其余为0。

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
print(y_onehot.shape)
print(y[0])
print(y_onehot[0,:])


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# 前向传播函数
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    # 输入层每个样本加一个偏置数据(为了b)
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    # 隐藏层每层也加一个偏置数据b
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T

    h = sigmoid(z3)
    return a1, z2, a2, z3, h


# 代价函数
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # 分割得出第一次的参数
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)],(hidden_size, (input_size + 1))))
    # 分割得出第二次的参数
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):],(num_labels, (hidden_size + 1))))
    # 得到前向传播函数的中间及结果数据
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 计算损失值
    J = 0
    # 累加每一个样本的cost
    for i in range(m):
        # 计算样本i的cost
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply(1 - y[i,:],np.log(1-h[i,:]))
        J += np.sum(first_term - second_term)
        # print(J)
    J = J / m

    # 正则化代价函数,注意不需要对b
    J += (float(learning_rate / (2 * m))) * (np.sum(np.power(theta1[:, 1:],2)) + np.sum(np.power(theta2[:, 1:], 2)))
    return J


# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# 随机初始化完整网络参数大小的参数数组
# 取值范围：-0.125 ~ 0.125
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
print(params)
m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)

# 将参数数组解开为每个层的参数矩阵
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, hidden_size + 1)))

# print(theta1.shape)
# print(theta2.shape)

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
# print(a1.shape)
# print(z2.shape)
# print(a2.shape)
# print(z3.shape)
# print(h.shape)
m_cost = cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(m_cost)


# sigmoid的梯度
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# 反向传播计算梯度
# 由于 反向传播所需的计算 是 代价函数中所需的计算过程，
# 我们实际上将 扩展 代价函数 以执行 反向传播 并 返回 代价 和 梯度 。
def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)],(hidden_size, input_size + 1)))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):],(num_labels, hidden_size + 1)))

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    J = 0
    deltal1 = np.zeros(theta1.shape)
    deltal2 = np.zeros(theta2.shape)

    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply(1-y[i,:],np.log(1-h[i,:]))
        J += np.sum(first_term - second_term)

    # print(J)
    J = J / m
    # print(J)
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:],2))+ np.sum(np.power(theta2[:,1:],2)))


    # 反向传播
    for t in range(m):
        a1t = a1[t,:] # （1，401）
        z2t = z2[t,:] # (1,25)
        a2t = a2[t,:] # （1,26）
        ht = h[t,:] # (1,10)
        yt = y[t,:] # (1,10)

        d3t = ht - yt # (1,10)

        z2t = np.insert(z2t, 0, values=np.ones(1))# (1,26)
        d2t = np.multiply(d3t * theta2, sigmoid_gradient(z2t))# (1,26)

        # 累加θ1一个样本所有列数据的梯度
        deltal1 = deltal1 + (d2t[:,1:]).T * a1t
        # 累加θ2的梯度
        deltal2 = deltal2 + d3t.T * a2t
    # 得到一个样本的平均梯度
    deltal1 = deltal1 / m
    deltal2 = deltal2 / m
    # 添加正则化项
    deltal1[:,1:] = deltal1[:,1:] + (theta1[:,1:] * learning_rate) / m
    deltal2[:,1:] = deltal2[:,1:] + (theta2[:,1:] * learning_rate) / m

    # 连接两个θ参数为一个，并铺展为1维
    grad = np.concatenate((np.ravel(deltal1), np.ravel(deltal2)))


    # 返回损失值和参数的梯度值
    return J, grad


# 不要传错y,应该要传 y_onehot
J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(J)
print(grad.shape)


# 以上建成了我们的训练网络，接下来进行训练得到参数
from scipy.optimize import minimize
# jac：可选，目标函数的雅可比矩阵（目标函数的一阶偏导）。
# jac可传入梯度的计算函数，如果设为True，则目标函数除了要返回cost之外，还要返回梯度grad值
# options参数，可选，字典类型，算法的其他选项。
# maxiter : int
# 算法的最大迭代次数
# disp : bool
# disp=True时打印收敛信息
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
print(fmin)

# 从函数调用返回结果提取训练后得到的参数
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, hidden_size + 1)))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
# 每一个样本行中概率值最大（每一行的列的最大值）的位置即为该样本最终的分类，由于下标与类差1，因此+1
y_pred = np.array(np.argmax(h,axis=1) + 1)
print(y_pred)

# 计算精确度
# 注意这里用的是y，不是y_onehot
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = sum(map(int, correct)) / len(correct)
print(f'accuracy: {accuracy * 100}%')