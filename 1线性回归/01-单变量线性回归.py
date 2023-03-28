import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('../data_sets/mydata.txt',header=None,names =[ 'Population','Profit'])
# head()方法用来查看前几行数据（默认是前5行），不过也可以通过括号内的参数指定前几行
# print(data.head())
# describe()方法用于获取数据集的常用统计量信息，包括计数、平均数、标准差、最大值、最小值及4分位差。需要注意的是，该方法仅会返回数值型变量的信息
# 四分位数是将一组数据由小到大（或由大到小）排序后，用3个点将全部数据分为4等份，与这3个点位置上相对应的数值称为四分位数
# print(data.describe())
# x:数据框列的标签或位置参数  y:数据框行的标签或位置参数 kind:图像类型（scatter表示散点图）,figsize:元组形式，表示图片尺寸大小
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
# 显示图像
# plt.show()
# 代价函数
def computeCost(X, y, theta):
    # print(theta.T)
    # theta.T返回矩阵的转置矩阵
    # 矩阵乘法和减法
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
# insert()参数介绍：
# loc:  int型，表示第几列；若在第一列插入数据，则 loc=0
# column: 给插入的列取名，如 column='新的一列'
# value：数字，array，series等都可
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
# shape[1]返回列数，shape[0]返回行数，shape返回元组（行数，列数）
cols = data.shape[1]
# iloc[ : , : ]
# 前面的冒号就是取行数，后面的冒号是取列数
X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
# print(X.head())
y = data.iloc[:,cols-1:cols]#X是所有行，最后一列
# print(y.head())
# 代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。
X = np.matrix(X.values)
y = np.matrix(y.values)
# 我们还需要初始化theta。
# np.array([0,0])返回一个数组

theta = np.matrix(np.array([0,0]))
# print(theta)
print("X:",X.shape)
# print(y.shape)
# print(theta.shape)
print(computeCost(X, y, theta))
# batch gradient decent（批量梯度下降）
def gradientDescent(X, y, theta, alpha, iters):
    # np.zeros((r,l)),生成r行l列的值全为0（或0.）的矩阵
    temp = np.matrix(np.zeros(theta.shape))
    print("temp：",temp.shape)
    # 获取theta元素个数
    # ravel()多维降为一维a
    parameters = int(theta.ravel().shape[1])
    # 生成元素数量为迭代次数的，值为0的numpy矩阵
    cost = np.zeros(iters)
    for i in range(iters):
        print("结果：",(X*theta.T).shape)
        error = (X * theta.T) - y   # 计算每一个f(x)-y
        # print(error.shape)  # (47, 1)
        # print(theta.shape)  # (1, 3)
        # print(X.shape)  # (47, 3)
        for j in range(parameters):
            # multiply  数组对应元素位置相乘(求内积)
            #1、当两个矩阵规格大小一样时，得到结果则是两个矩阵的内积
            #2、当两个矩阵大小不一样，则先将小的扩大到与另一矩阵大小一样，再求内积
            # y=θ0+θ1*x，j=0时，处理θ0，求和不需要*x,X[:, j]的值全为1，*了相当于没*
            # j=1时，处理θ1，X[:, j]的值为样本x值
            term = np.multiply(error, X[:, j])
            # 同步更新参数的值
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
            # print(i,j,temp)
        # 更新theta的值
        theta = temp
        # 计算此时当前theta值的代价函数
        cost[i] = computeCost(X, y, theta)
    return theta, cost
# 学习速率
alpha = 0.45
# 迭代次数
iters = 1000
g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g)
print(computeCost(X,y,g))
# 绘制预测函数图像(同时含有样本散点图)
# linspace:作用：在指定的大间隔内（start，stop），返回固定间隔的数据。他们返回num个等间距的样本(array形式)
x = np.linspace(data.Population.min(), data.Population.max(), 100)
# 利用梯度下降得到的参数构造函数
f = g[0, 0] + (g[0, 1] * x)
# subplots返回的值的类型为元组，其中包含两个元素：第一个为一个画布，第二个是子图
fig, ax = plt.subplots(figsize=(12,8))
# 'r':表示颜色，红色；label：图例，legend文字
ax.plot(x, f, 'r', label='Prediction')
# scatter:绘制散点图.x, y位置
ax.scatter(data.Population, data.Profit, label='Traning Data')
# legend:在坐标系中放置图例,loc:表示位置,2代表‘upper left’
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

# 绘制梯度下降的代价函数
fig, ax = plt.subplots(figsize=(12,8))
# arrange函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]
print(cost)
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

# 使用scikit-learn的线性回归函数，与我们自己拟合的效果进行对比
from sklearn import linear_model
model = linear_model.LinearRegression()
# print(model.fit(X, y))



