import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
"""
设想你是工厂的生产主管，你有一些芯片在两次测试中的测试结果。
对于这两次测试，你想决定是否芯片要被接受或抛弃。
为了帮助你做出艰难的决定，你拥有过去芯片的测试数据集，
从其中你可以构建一个逻辑回归模型。
"""
path = "../data_sets/ex2data2.txt"
data2 = pd.read_csv(path, header=None, names=["Test 1", "Test 2", "Accepted"])
# print(data2.head())

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]
# 画出散点图
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.scatter(positive["Test 1"], positive["Test 2"], s=50, c='b', marker='o', label='Accepted')
ax1.scatter(negative["Test 1"], negative["Test 2"], s=50, c='r', marker='x', label='Rejected')
ax1.legend()
ax1.set_xlabel('Test 1 Score')
ax1.set_ylabel('Test 2 Score')
# plt.show()

# 创建并添加一组多项式特征
# x1^i*x2^j
degree = 5
x1 = data2["Test 1"]
x2 = data2["Test 2"]
data2.insert(3, 'Ones', 1)
for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i - j) * np.power(x2, j)
# drop()函数的功能是通过指定的索引或标签名称，也就是行名称或者列名称进行删除数据。
# 指定删除相关的列，没有带columns，所以要指出是哪个轴上的
# inplace，设置是否在原数据上进行操作。
# 删除x1,x2的值，因为上面已经包括了这两个变量的值
data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)
# print(data2.head())


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
# learningRate:λ
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    # 注意不需要对b正则化
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


# 代价函数对参数求导的表达式
def gradienReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if 0 == i:
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term)) / len(X) + (learningRate / len(X)) * theta[:, i]
    return grad


# 初始化变量
cols = data2.shape[1]
X2 = data2.iloc[:, 1:cols]
y2 = data2.iloc[:, [0]]
# 将类型转为pd.DataFrame转为np.array类型
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(data2.shape[1]-1)
learingRate = 1
# 计算当参数值全为0时的cost值
print(costReg(theta2,X2,y2,learingRate))

# 利用优化函数得到cost最小的theta参数
result = opt.fmin_tnc(func=costReg,x0=theta2,fprime=gradienReg,args=(X2,y2,learingRate))
print(result)
theta_min = np.matrix(result[0])


# 预测函数
def predict(theta,X):
    theta = np.matrix(theta)
    X = np.matrix(X)
    probability = sigmoid(X*theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


predictions = predict(theta_min,X2)
# print(predictions)
correct = [1 if (1 == a and 1 == b) or (0 == a and 0 == b) else 0 for (a ,b) in zip(predictions, y2)]
accuracy = sum(map(int, correct)) / len(correct)
print(f"accuracy = {accuracy*100}%")


# 利用高级python库scikit-learn解决问题(只需要传数据参数即可)
# 调用sklearn的线性回归包
from sklearn import linear_model
# penalty：选择哪一种正则化方式：l1:惩罚项（正则项）为参数绝对值之和；l2:惩罚项（正则项）为参数平方和的开方
# C ：正则化强度（超参数）的倒数
model = linear_model.LogisticRegression(penalty ='l2',C=1.0)
model.fit(X2,y2.ravel())
print(model.score(X2,y2))

