import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sb


# 找出每个样本离得最近的聚类中心， 返回每个样本对应的聚类中心
def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum(np.power(X[i, :] - centroids[j, :], 2))
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx


# 测试一下
data = loadmat('../data_sets/ex7data2.mat')
# print(data)
X = data['X']
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = find_closest_centroids(X, initial_centroids)
print(idx[0:3])

data2 = pd.DataFrame(data['X'], columns=['X1', 'X2'])
# print(data2.head())
# 设置背景色、风格、字型、字体等。
sb.set(context='notebook', style='white')
sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
plt.show()


# 计算簇的聚类中心
def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()

    return centroids


# k=3
# print(compute_centroids(X, idx, 3))


# k-means 算法实现
def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids


idx, centroids = run_k_means(X, initial_centroids, 10)

# ret = np.where(0 == idx)
# print(ret)
# 后面不加[0]最后出来的是三维数组
cluster1 = X[np.where(0 == idx)[0], :]
# print(cluster1)
cluster2 = X[np.where(1 == idx)[0], :]
cluster3 = X[np.where(2 == idx)[0], :]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster2')
ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster3')
ax.legend()
plt.show()


# 随机选择初始化聚类中心
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    # 产生随机坐标（X数据集范围内）
    # lb, ub, size
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids


# print(init_centroids(X, 3))

# 将k-means应用于图片压缩
# 我们可以使用聚类来找到最具代表性的少数颜色，
# 并使用聚类分配将原始的24位颜色映射到较低维的颜色空间。


# 显示图片
# 下面两行代码在pycharm中显示不出图片
# from IPython.display import Image, display
# display(Image(filename='../data_sets/bird_small.png'))
path = '../data_sets/bird_small.png'
image = plt.imread(path)
plt.imshow(image)
plt.show()

# 加载图片数据
image_data = loadmat('../data_sets/bird_small.mat')
print(image_data)
A = image_data['A']
# 三维数据， 行像素数量，列像素数量， RGB参数
print(A.shape)  # （128， 128， 3）

# 标准化
A = A / 255.
# print(A)

X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
print(X.shape)  # (16384, 3)
initial_centroids = init_centroids(X, 16)

idx, centroids = run_k_means(X, initial_centroids, 10)
# 最后一次获得各元素对应的最近的聚类中心
idx = find_closest_centroids(X, centroids)
# map each pixel to the centroid value
# 将idx从浮点数转为整数后再传入
# 最终每一个RGB的值都只来自centroids中，一共16种取值
X_recovered = centroids[idx.astype(int),:]
print(X_recovered)
print(X_recovered.shape)
# 恢复为原来的三维
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
print(X_recovered.shape)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[1].imshow(X_recovered)
# 您可以看到我们对图像进行了压缩，但图像的主要特征仍然存在。 这就是K-means
plt.show()

# 下面我们来用scikit-learn来实现K-means。
from skimage import io
# cast to float, you need to do this otherwise the color would be weird after clustring
path = '../data_sets/bird_small.png'
pic = io.imread(path) / 255.
io.imshow(pic)
plt.show()
print(pic.shape)

# 序列化数据
data = pic.reshape(pic.shape[0] * pic.shape[1], pic.shape[2])
print(data.shape)


# 导入kmeans库
from sklearn.cluster import KMeans
"""
n_clusters：整形，缺省值=8 【生成的聚类数，即产生的质心（centroids）数。】
max_iter：整形，缺省值=300
            执行一次k-means算法所进行的最大迭代数。
n_init：整形，缺省值=10
        用不同的质心初始化值运行算法的次数，最终解是在inertia意义下选出的最优结果。
init：有三个可选值：’k-means++’， ‘random’，或者传递一个ndarray向量。
        此参数指定初始化方法，默认值为 ‘k-means++’。
n_jobs：整形数。　指定计算所用的进程数。内部原理是同时进行n_init指定次数的计算。
        （１）若值为 -1，则用所有的CPU进行运算。若值为1，则不进行并行运算，这样的话方便调试。
        （２）若值小于-1，则用到的CPU数为(n_cpus + 1 + n_jobs)。因此如果 n_jobs值为-2，则用到的CPU数为总CPU数减1。
"""
# 不要开多进程，否则会重复执行上面的代码
model = KMeans(n_clusters=16, n_init=100, n_jobs=1)
"""
fit(X[,y]):
　计算k-means聚类。
fit_predictt(X[,y]):
　计算簇质心并给每个样本预测类别。
"""
print(model.fit(data))

centroids = model.cluster_centers_
print(centroids.shape)
C = model.predict(data)
print(C)
print(C.shape)
print(centroids[C].shape)

compressed_pic = centroids[C].reshape((128, 128, 3))
fig, ax = plt.subplots(1 ,3)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
ax[2].imshow(X_recovered)
plt.show()






