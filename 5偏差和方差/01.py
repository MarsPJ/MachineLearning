import numpy as np
import scipy.optimize as opt
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
在练习的前半部分，您将使用水库水位的变化实现正则化线性回归来预测大坝的出水量。
在下半部分中，您将对调试学习算法进行诊断，并检查偏差和方差的影响。
本次的数据是以.mat格式储存的，x表示水位的变化，y表示大坝的出水量。
数据集共分为三部分：训练集（X, y）、交叉验证集（Xval, yval）和测试集（Xtest, ytest）。
"""


def load_data():
    d = sio.loadmat('../data_sets/ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])


X, y, Xval, yval, Xtest, ytest = load_data()
# print(X)
# print(y)
# print(Xval)
# print(yval)
# print(Xtest)
# print(ytest)
# print(X.shape)
# print(y.shape)
# print(Xval.shape)
# print(yval.shape)
# print(Xtest.shape)
# print(ytest.shape)
df = pd.DataFrame({'water_level': X, 'flow': y})
print(df.head())
"""
Seaborn是基于matplotlib的Python可视化库。
 它提供了一个高级界面来绘制有吸引力的统计图形。
 Seaborn其实是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易，
 不需要经过大量的调整就能使你的图变得精致。
 但应强调的是，应该把Seaborn视为matplotlib的补充，而不是替代物。
"""
# fig_reg: True绘制回归曲线，
# size:子图的高度
sns.lmplot('water_level', 'flow', data=df, fit_reg=False, size=5)
plt.title("water_level-flow")
plt.legend(loc='lower right')
plt.show()
# 添加偏置项
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]
# print(X.shape)
# print(y.shape)
print("aaa", Xval.shape)


# print(yval.shape)
# print(Xtest.shape)
# print(ytest.shape)


# 代价函数
def cost(theta, X, y):
    """
    X: R(m*n), m records, n features
    y: R(m)
    theta : R(n), linear regression parameters
    """
    # 化为矩阵计算，最后cost是一个只有一个元素的二维的矩阵
    # X = np.matrix(x)
    # theta = np.matrix(theta)
    # y = np.matrix(y)
    # print(X.shape)
    # print(y.shape)
    # print(theta.shape)
    # m = X.shape[0]
    # inner = np.matmul(X, theta.T) - y.T
    # # 直接用矩阵乘法就能又得到平方又求和
    # square_sum = np.matmul(inner.T, inner)
    # cost = square_sum / (2 * m)

    # 原版方法，cost是一个数
    m = X.shape[0]

    inner = X @ theta - y  # R(m*1)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost


# 只有一个特征，设置两个参数（参数1和偏置b）
theta = np.ones(X.shape[1])
m_cost = cost(theta, X, y)
print(m_cost)


# 正则化代价函数
def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = (l / (2 * m)) * np.sum(np.power(theta[1:], 2))
    return cost(theta, X, y) + regularized_term


# 梯度
def gradient(theta, X, y):
    m = X.shape[0]
    # 只做向量计算
    inner = X.T @ (X @ theta - y)
    return inner / m


grad = gradient(theta, X, y)
print(grad)


# 正则化梯度
def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    # 复制一份theta的数据
    regularized_term = theta.copy()
    # 偏置项不需要正则化
    regularized_term[0] = 0
    regularized_term = (l / float(m)) * regularized_term
    return gradient(theta, X, y) + regularized_term


grad_reg = regularized_gradient(theta, X, y)
print(grad_reg)


# 拟合数据
def linear_regression_np(X, y, l=1):
    theta = np.ones(X.shape[1])

    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': False})
    return res


theta = np.ones(X.shape[1])
# λ = 0不进行正则化
# print(linear_regression_np(X, y, 0))
final_theta = linear_regression_np(X, y, l=0).x
b = final_theta[0]
m = final_theta[1]

plt.scatter(X[:, 1], y, label='Training data')
plt.plot(X[:, 1], X[:, 1] * m + b, label='Prediction')
plt.legend(loc='upper left')
plt.show()

# 1.使用训练集的子集来拟合模型
# 2.在计算训练代价和交叉验证代价时，没有用正则化l=0
# 3.记住使用相同的训练自己来计算训练代价
training_cost, cv_cost = [], []
m = X.shape[0]
# 不断增加训练集，得到不同数量训练集时的训练代价和交叉验证代价
for i in range(1, m + 1):
    res = linear_regression_np(X[:i, :], y[:i], l=0)
    tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
    cv = regularized_cost(res.x, Xval, yval, l=0)
    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
plt.legend(loc='upper right')
plt.title('no regularization')
plt.show()


# 结果欠拟合，偏差太大，由于模型并没有正则化，因此只能通过增加特征解决


# 创建多项式特征
def poly_features(x, power, as_ndarray=False):
    # data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    data = {'{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)
    # 由参数选择返回经过幂处理后的数据形式（数组/DataFrame）
    # as_matrix：Convert the frame to its Numpy-array representation.
    return df.as_matrix() if as_ndarray else df


# 归一化处理
def normalize_feature(df):
    # 在Pandas中，DataFrame和Series等对象需要执行批量处理操作时，可以借用apply()函数来实现。
    # apply()的核心功能是实现“批量”调度处理，至于批量做什么，由用户传入的函数决定（自定义或现成的函数）。
    # 函数传递给apply()，apply()会帮用户在DataFrame和Series等对象中（按行或按列）批量执行传入的函数。
    # func: 应用于每一列或每一行的函数，这个函数可以是Python内置函数、Pandas或其他库中的函数、自定义函数、匿名函数。
    # axis: 设置批处理函数按列还是按行应用，0或index表示按列应用函数，1或columns表示按行应用函数，默认值为0。
    # raw: 设置将列/行作为Series对象传递给函数，还是作为ndarray对象传递给函数。raw是bool类型，默认为False。
    return df.apply(lambda colum: (colum - colum.mean()) / colum.std())


# 扩展多项式特征的总函数：归一化并且增加为截距服务的1
# 返回的是矩阵
def prepare_poly_data(*args, power):
    def prepare(x):
        # 扩展特征值
        df = poly_features(x, power=power)
        # 归一化
        ndarr = normalize_feature(df).as_matrix()
        # 增加为截距服务的1
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]


print(X.shape)  # (12,2)
print(type(X))
# 重新加载原始数据
X, y, Xval, yval, Xtest, ytest = load_data()
print(X.shape)  # (12,)
print(type(X))
# 测试一下
# 注意这里调用的是poly_features不是prepare_poly_data
temp = poly_features(X, power=3)
print(type(temp))
print(temp.shape)
print(temp.head())

# 扩展特征到8阶
X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)
print(X_poly[:3, :])


# 画出学习曲线（没有正则化）
# 作图
def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1, m + 1):
        res = linear_regression_np(X[:i, :], y[:i], l=l)
        # 注意训练代价和交叉验证代价传入的X，y的值和数量都不同
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc='upper right')


plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
plt.title('add features no regularization power=8')
plt.show()  # training cost为0，结果过拟合

# 接下来进行正则化
# λ = 1
plot_learning_curve(X_poly, y, Xval_poly, yval, l=1)
plt.title('add features power=8 regularization lambda=1')
plt.show()  # training cost增加了，减轻了过拟合
# λ = 100
plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)
plt.title('add features power=8 regularization lambda=100')
plt.show()  # train cost过大，欠拟合了

# 找最佳的λ
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []

for l in l_candidate:
    res = linear_regression_np(X_poly, y, l)

    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)

    # 前面的是收集不同的数据量的tc和cv
    # 现在是同数据量，不同λ
    training_cost.append(tc)
    cv_cost.append(cv)


plt.plot(l_candidate, training_cost, label='training')
plt.plot(l_candidate, cv_cost, label='cross validation')

plt.legend(loc='upper left')
plt.xlabel('lambda')
plt.ylabel('cost')
plt.title("lambda - training cost & cross validation cost")
plt.show()

# 由图知training cost在不断增加，因此选cv cost最小的λ
l_min = l_candidate[np.argmin(cv_cost)]
print(l_min)

# 用测试集计算不同λ值时,test cost
for l in l_candidate:
    theta = linear_regression_np(X_poly, y, l).x
    # 由于是观测测试样本拟合效果，因此不需要得到正则化后的cost，而是真实的cost
    print(f'test cost(l={l}) = {cost(theta, Xtest_poly, ytest)}')

# 调参后，λ=0.3是最优选择，这时候测试代价最小

