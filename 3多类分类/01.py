# 我们将使用逻辑回归来识别手写数字（0到9）。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Scipy是基于Numpy构建的一个集成了多种数学算法和方便的函数的Python模块。
from scipy.io import loadmat

data = loadmat("../data_sets/ex3data1.mat")
print(data)
# 可视化数据集
raw_X = data['X']
raw_y = data['y']


# 画一张图片
def plot_an_image(X):
    pick_one = np.random.randint(5000)
    # pick_one = 4998
    # 读取该图片像素信息
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(10, 10))
    # 先将像素信息还原成20*20的矩阵，再转置
    ax.imshow(image.reshape(20, 20).T, cmap='gray_r')
    # 去掉坐标轴
    plt.xticks([])
    plt.yticks([])
    plt.show()


plot_an_image(raw_X)


# 随机画100张图片
def plot_100_images(X):
    # 随机选取数据集里的100个数据
    sample_index = np.random.choice(len(X), 100)
    images = X[sample_index, :]
    # 定义10*10的子绘图区
    fig, ax = plt.subplots(ncols=10, nrows=10, figsize=(8, 8), sharex=True, sharey=True)
    # 在每一个子绘图区中画出一个数字
    for r in range(10):
        for c in range(10):
            # 两层循环是为了方便选择ax子绘图区，否则直接range(100)即可（images中的数据是连续存放的）
            ax[r, c].imshow(images[10 * r + c].reshape(20, 20).T, cmap='gray_r')
    # 去掉坐标轴
    plt.xticks([])
    plt.yticks([])
    plt.show()


plot_100_images(raw_X)
"""
图像在martix X中表示为400维向量（其中有5,000个）。 
400维“特征”是原始20 x 20图像中每个像素的灰度强度。
类标签在向量y中作为表示图像中数字的数字类。
"""


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


# 向量化梯度函数
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    # print("error: ", error.shape)# (5000 ,1)
    # print("X: ", X.shape)# (5000, 401)
    grad = ((X.T * error) / len(X)).T + (learningRate / len(X)) * theta
    # 截距b不需要正则化
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    return np.array(grad).ravel()


from scipy.optimize import minimize


# 将y从类标签转换为每个分类器的二进制值（要么是类i，要么不是类i）,
# 即分为10个类，是该类则为1，否则为0
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k * (n+1)矩阵,k是分类器数量，n代表数据中x1,...,xn，因为参数比n多一个 b
    all_theta = np.zeros((num_labels, params + 1))
    # 在X前插入一列全一数据，为了方便进行矩阵运算（因为多了个b）
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        # 考虑第i个分类器的情况，如果是第i个分类器,则y值为1，否则为0
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = y_i.reshape((rows, 1))
        # 此时对于当前第i个分类器函数x=X,y=y_i
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        # minimize返回优化的结果（参数的最优值）
        all_theta[i - 1, :] = fmin.x
    return all_theta


# 此时的data是一个字典
rows = data['X'].shape[0]
params = data['X'].shape[1]

all_theta = np.zeros((10, params + 1))

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
"""
注意，theta是一维数组，因此当它被转换为计算梯度的代码中的矩阵时，它变为（1×401）矩阵。 
"""
theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = y_0.reshape((rows, 1))
print(X.shape)
print(y_0.shape)
print(theta.shape)
print(all_theta.shape)

# 我们还检查y中的类标签，以确保它们看起来像我们想象的一致。
print(np.unique(data['y']))
all_theta = one_vs_all(data['X'], data['y'], 10, 1)


# print(all_theta)


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    h = sigmoid(X * all_theta.T)

    #  Returns the indices of the maximum values along an axis.1表示行方向
    h_argmax = np.argmax(h, axis=1)

    #
    h_argmax = h_argmax + 1
    return h_argmax


y_pred = predict_all(data['X'], all_theta)
print(y_pred)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = sum(map(int, correct)) / len(correct)
print(f"accuracy: {accuracy * 100}%")
