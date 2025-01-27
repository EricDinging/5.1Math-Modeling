# 导入Sklearn相关包
import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets._samples_generator import make_blobs
# 导入matplotlib相关包
import matplotlib.pyplot as plt
from collections import Counter
import copy

# 自定义K-Means方法
class KMeans():
    # 定义数据集，簇中心数量，最大迭代次数
    def __init__(self, k=3, max_iter=300):
        self.k = k
        self.max_iter = max_iter

    # 定义距离（此处采用欧式距离）
    def dist(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    # 得到每个样本的label
    def get_label(self, x):
        min_dist_with_mu = 999999
        label = -1

        for i in range(self.mus_array.shape[0]):
            dist_with_mu = self.dist(self.mus_array[i], x)
            if min_dist_with_mu > dist_with_mu:
                min_dist_with_mu = dist_with_mu
                label = i

        return label

    # 得到簇
    def get_mu(self, X):
        index = np.random.choice(X.shape[0], 1, replace=False)
        mus = []
        mus.append(X[index])
        for _ in range(self.k - 1):
            max_dist_index = 0
            max_distance = 0
            for j in range(X.shape[0]):
                min_dist_with_mu = 999999

                for mu in mus:
                    dist_with_mu = self.dist(mu, X[j])
                    if min_dist_with_mu > dist_with_mu:
                        min_dist_with_mu = dist_with_mu

                if max_distance < min_dist_with_mu:
                    max_distance = min_dist_with_mu
                    max_dist_index = j
            mus.append(X[max_dist_index])

        mus_array = np.array([])
        for i in range(self.k):
            if i == 0:
                mus_array = mus[i]
            else:
                mus[i] = mus[i].reshape(mus[0].shape)
                mus_array = np.append(mus_array, mus[i], axis=0)

        return mus_array

    # 选择初始簇
    def init_mus(self):
        for i in range(self.mus_array.shape[0]):
            self.mus_array[i] = np.array([0] * self.mus_array.shape[1])

    # 进行分类后计算新的簇
    def fit(self, X):
        self.mus_array = self.get_mu(X)
        iter = 0

        while(iter < self.max_iter):

            old_mus_array = copy.deepcopy(self.mus_array)

            Y = []
            # 将X归类
            for i in range(X.shape[0]):
                y = self.get_label(X[i])
                Y.append(y)

            self.init_mus()
            # 将同类的X累加
            for i in range(len(Y)):
                self.mus_array[Y[i]] += X[i]

            count = Counter(Y)
            # 计算新的mu
            for i in range(self.k):
                self.mus_array[i] = self.mus_array[i] / count[i]

            diff = 0
            for i in range(self.mus_array.shape[0]):
                diff += np.linalg.norm(self.mus_array[i] - old_mus_array[i])
            if diff == 0:
                break
            iter += 1

        self.E = 0
        for i in range(X.shape[0]):
            self.E += self.dist(X[i], self.mus_array[Y[i]])
        print('E = {}'.format(self.E))
        return np.array(Y)


# 随机生成数据，利用K-Means方法进行聚类，实现可视化
if __name__ == '__main__':

    fig = plt.figure(1)

    plt.subplot(221)
    center = [[1, 1], [-1, -1], [1, -1]]
    cluster_std = 0.35
    X1, Y1 = make_blobs(n_samples=1000, centers=center,
                        n_features=3, cluster_std=cluster_std, random_state=1)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

    plt.subplot(222)
    km1 = KMeans(k=3)
    km_Y1 = km1.fit(X1)
    mus = km1.mus_array
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=km_Y1)
    plt.scatter(mus[:, 0], mus[:, 1], marker='^', c='r')

    plt.subplot(223)
    X2, Y2 = make_moons(n_samples=1000, noise=0.1)
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2)

    plt.subplot(224)
    km2 = KMeans(k=2)
    km_Y2 = km2.fit(X2)
    mus = km2.mus_array
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=km_Y2)
    plt.scatter(mus[:, 0], mus[:, 1], marker='^', c='r')
    plt.show()
