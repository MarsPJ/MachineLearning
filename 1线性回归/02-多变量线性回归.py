import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
path="data_sets/ex1data2.txt"
data=pd.read_csv(path,header=None,names=['Size','Bedrooms','price'])
# print(data.head())
# 特征归一化
data=(data-data.mean())/data.std()
# print(data.head())
data.insert(0,'Ones',1)
cols=data.shape[1]
# 解析数据集,得到X集合和y集合
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

# 将X和y集合转为numpy矩阵
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0,0]))
def computeCost(X,y,theta):
    inner=np.power((X*theta.T-y),2)
    return np.sum(inner)/(2*len(X))

def gradientDescent(X,y,theta,alpha,iters):
    parameters=int(theta.ravel().shape[1])
    costs=np.zeros(iters)
    # 必须先处理好theta,才能参与运算
    temp=np.matrix(np.zeros(theta.shape))
    for i in range(iters):
        error = (X * theta.T) - y
        # print(error.shape)#(47, 1)
        # print(theta.shape)#(1, 3)
        # print(X.shape)#(47, 3)
        for j in range(parameters):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j]-(alpha/len(X))*np.sum(term)
        theta=temp
        costs[i]=computeCost(X,y,theta)
    return theta,costs
alpha=0.01
iters=1000
g,costs=gradientDescent(X,y,theta,alpha,iters)
print(computeCost(X,y,g))
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),costs,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Costs')
ax.set_title('Error vs. Training Epoch')
plt.show()
