import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_features = 2, n_informative = 2, n_redundant = 0, n_samples = 1000, n_classes = 2, random_state = 0)
y_classes = np.unique(y)
y[y == y_classes[0]] = -1
y[y == y_classes[1]] = 1
def cost(X, y, W):
    """
    对数几率回归的代价函数
    args:
        X - 训练数据集
        y - 目标标签值
        W - 权重系数
    return:
        代价函数值
    """
    power = -np.multiply(y, X.dot(W))
    p1 = power[power <= 0]
    p2 = -power[-power < 0]
    # 解决 python 计算 e 的指数幂溢出的问题
    return np.sum(np.log(1 + np.exp(p1))) + np.sum(np.log(1 + np.exp(p2)) - p2)
def dcost(X, y, W):
    """
    对数几率回归的代价函数的梯度
    args:
        X - 训练数据集
        y - 目标标签值
        W - 权重系数
    return:
        代价函数的梯度
    """
    return X.T.dot(np.multiply(-y, 1 / (1 + np.exp(np.multiply(y, X.dot(W))))))
def ddcost(X, y, W):
    """
    对数几率回归的代价函数的黑塞矩阵
    args:
        X - 训练数据集
        y - 目标标签值
        W - 权重系数
    return:
        代价函数的黑塞矩阵
    """
    exp = np.exp(np.multiply(y, X.dot(W)))
    result = np.multiply(exp, 1 / np.square(1 + exp))
    X_r = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_r[:, i] = np.multiply(result, X[:, i])
    return X_r.T.dot(X)

def direction(d, H):
    """
    更新的方向
    args:
        d - 梯度
        H - 黑塞矩阵
    return:
        更新的方向
    """
    return - np.linalg.inv(H).dot(d)
import numpy as np

def logisticRegressionSGD(X, y, max_iter=100, tol=1e-4, step=1e-1):
    W = np.zeros(X.shape[1])
    Ws = []
    Ws.append(W)
    xy = np.c_[X.reshape(X.shape[0], -1), y.reshape(X.shape[0], 1)]
    for it in range(max_iter):
        s = step / (np.sqrt(it + 1))
        np.random.shuffle(xy)
        X_new, y_new = xy[:, :-1], xy[:, -1:].ravel()
        for i in range(0, X.shape[0]):
            d = dcost(X_new[i], y_new[i], W)
            if (np.linalg.norm(d) <= tol):
                break
            W = W - s * d
            Ws.append(W)
        else:
            continue
    return Ws


y_classes = np.unique(y)
y[y == y_classes[0]] = -1
y[y == y_classes[1]] = 1
X_b = np.c_[np.ones((X.shape[0], 1)), X]
Ws = logisticRegressionSGD(X_b, y)
W = Ws[len(Ws) - 1]
print(W)
import numpy as np
import matplotlib.pyplot as plt



plt.rcParams['font.sans-serif'] = ['simsun']  # 选择一个本地的支持中文的字体
fig, ax = plt.subplots()
ax.set_facecolor('#f8f9fa')

costs = []
for i in range(len(Ws)):
    costs.append(cost(X_b, y, Ws[i]))

ax.plot(np.arange(0, len(Ws), 1), costs)

ax.set_title('对数几率回归-随机梯度下降法', color='#264653')
ax.set_xlabel('迭代次数', color='#264653')
ax.set_ylabel('代价函数', color='#264653')
ax.tick_params(labelcolor='#264653')
plt.show()

def logisticRegressionSVRG(X, y, max_iter=100, m = 100, tol=1e-4, step=1e-1):
    W = np.zeros(X.shape[1])
    Ws = []
    Ws.append(W)
    for it in range(max_iter):
        s = step / (np.sqrt(it + 1))
        g = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            g = g + dcost(X[i], y[i], W)
        g = g / X.shape[0]
        w = W
        for it in range(m):
            i = np.random.randint(0, X.shape[0])
            d_w = dcost(X[i], y[i], w)
            d_W = dcost(X[i], y[i], W)
            d = d_w - d_W + g
            if (np.linalg.norm(d) <= tol):
                break
            w = w - s * d
        W = w
        Ws.append(W)
    return Ws

y_classes = np.unique(y)
y[y == y_classes[0]] = -1
y[y == y_classes[1]] = 1
X_b = np.c_[np.ones((X.shape[0], 1)), X]
Ws = logisticRegressionSVRG(X_b, y)
W = Ws[len(Ws) - 1]
print(W)


plt.rcParams['font.sans-serif'] = ['simsun']  # 选择一个本地的支持中文的字体
fig, ax = plt.subplots()
ax.set_facecolor('#f8f9fa')

costs = []
for i in range(len(Ws)):
    costs.append(cost(X_b, y, Ws[i]))

ax.plot(np.arange(0, len(Ws), 1), costs)

ax.set_title('对数几率回归-方差缩减随机梯度下降法', color='#264653')
ax.set_xlabel('迭代次数', color='#264653')
ax.set_ylabel('代价函数', color='#264653')
ax.tick_params(labelcolor='#264653')
plt.show()


def logisticRegressionSAGA(X, y, max_iter=100, tol=1e-4, step=1e-1):
    W = np.zeros(X.shape[1])
    Ws = []
    Ws.append(W)
    p = np.zeros(X.shape[1])
    d_prev = np.zeros(X.shape)
    for i in range(X.shape[0]):
        d_prev[i] = dcost(X[i], y[i], W)
    for it in range(max_iter):
        s = step / (np.sqrt(it + 1))
        for it in range(X.shape[0]):
            i = np.random.randint(0, X.shape[0])
            d = dcost(X[i], y[i], W)
            p = d - d_prev[i] + np.mean(d_prev, axis=0)
            d_prev[i] = d
            if (np.linalg.norm(p) <= tol):
                break
            W = W - s * p
        else:
            continue
        Ws.append(W)
    return Ws

import numpy as np

y_classes = np.unique(y)
y[y == y_classes[0]] = -1
y[y == y_classes[1]] = 1
X_b = np.c_[np.ones((X.shape[0], 1)), X]
Ws = logisticRegressionSAGA(X_b, y)
W = Ws[len(Ws) - 1]
print(W)

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['simsun']  # 选择一个本地的支持中文的字体
fig, ax = plt.subplots()
ax.set_facecolor('#f8f9fa')

costs = []
for i in range(len(Ws)):
    costs.append(cost(X_b, y, Ws[i]))

ax.plot(np.arange(0, len(Ws), 1), costs)

ax.set_title('对数几率回归-SAGA', color='#264653')
ax.set_xlabel('迭代次数', color='#264653')
ax.set_ylabel('代价函数', color='#264653')
ax.tick_params(labelcolor='#264653')
plt.show()