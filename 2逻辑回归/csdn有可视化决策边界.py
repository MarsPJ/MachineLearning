
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 1.读取
path = '../data_sets/ex2data2.txt'
data = pd.read_csv(path, names=['Exam1', 'Exam2', 'Accepted'])
data.head()

# 2.数据可视化
fig, ax = plt.subplots()
ax.scatter(data[data['Accepted'] == 0]['Exam1'], data[data['Accepted'] == 0]['Exam2'], c='r', marker='x', label='y=0')
ax.scatter(data[data['Accepted'] == 1]['Exam1'], data[data['Accepted'] == 1]['Exam2'], c='b', marker='o', label='y=1')
ax.legend()
ax.set_xlabel('exam1')
ax.set_ylabel('exam2')
plt.show()
# 从特征图看出来，这是线性不可分，下一步是特征映射
# 3.特征映射
def feature_mapping(x1, x2, power):
    data = {}
    for i in np.arange(power + 1):
        for j in np.arange(i + 1):
            data['F{}{}'.format(i - j, j)] = np.power(x1, i - j) * np.power(x2, j)
    return pd.DataFrame(data)


x1 = data['Exam1']
x2 = data['Exam2']
data2 = feature_mapping(x1, x2, 6)
print(data2.head())

# 4.构建数据集
X = data2.values
# (118, 28)
print(X.shape)
y = data.iloc[:, -1].values
y = y.reshape(len(y), 1)
# (118, 1)
print(y.shape)


# 损失函数
# 多项式需要正则化
# λ越小，容易过拟合；λ越大，容易欠拟合

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Cost_Function(X, y, theta, lr):
    A = sigmoid(X @ theta)
    first = y * np.log(A)
    second = (1 - y) * np.log(1 - A)
    reg = np.sum(np.power(theta[1:], 2)) * (lr / (2 * len(X)))
    return -np.sum(first + second) / len(X) + reg


theta = np.zeros((28, 1))
print(theta.shape)
lr = 1
cost_init = Cost_Function(X, y, theta, lr)
print(cost_init)


def gradientDescent(X, y, theta, alpha, iters, lr):
    costs = []
    for i in range(iters):
        reg = theta[1:] * (lr / len(X))
        reg = np.insert(reg, 0, values=0, axis=0)
        A = sigmoid(X @ theta)
        # X.T:X的转置
        theta = theta - (X.T @ (sigmoid(X @ theta) - y)) * alpha / len(X) - reg * alpha
        cost = Cost_Function(X, y, theta, lr)
        costs.append(cost)
        # if i % 1000 == 0:
        #     print(cost)
    return theta, costs


alpha = 0.001
iters = 200000
lamda = 0.01

theta_final, costs = gradientDescent(X, y, theta, alpha, iters, lamda)
print(costs)
print(theta_final)


# 准确率
def predict(X, theta):
    prob = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in prob]


y_ = np.array(predict(X, theta_final))
print(y_)
y_pre = y_.reshape(len(y_), 1)
# 求取均值
acc = np.mean(y_pre == y)
print(acc)
# 决策界面
x = np.linspace(-1.2, 1.2, 200)
xx, yy = np.meshgrid(x, x)
z = feature_mapping(xx.ravel(), yy.ravel(), 6).values
print(z.shape)
zz = z @ theta_final
zz = zz.reshape(xx.shape)
print(theta.shape)
fig, ax = plt.subplots()
ax.scatter(data[data['Accepted'] == 0]['Exam1'], data[data['Accepted'] == 0]['Exam2'], c='r', marker='x', label='y=0')
ax.scatter(data[data['Accepted'] == 1]['Exam1'], data[data['Accepted'] == 1]['Exam2'], c='b', marker='o', label='y=1')
ax.legend()
ax.set_xlabel('exam1')
ax.set_ylabel('exam2')
# plt.contour(xx, yy, zz, 0)
# plt.show()