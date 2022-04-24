# 导入Sklearn相关包
import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
# 导入matplotlib相关包
import matplotlib.pyplot as plt

# 读入数据并进行处理
# mac2id是一个字典：key是mac地址，value是对应mac地址的上网时长以及开始上网时间在onlinetimes中的索引
mac2id = dict()
# 用onlinetimes存（onlinetime,starttime)
onlinetimes = []
# 打开文件
f = open('/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-机器学习-聚类-DBSCAN算法与密度聚类【hxy】/internet.txt')
for line in f:
  # 获取一行中以','分割的第三个字符串，即mac地址
  mac = line.split(',')[2]
  # 获取上网时长
  onlinetime = int(line.split(',')[6])
  # 获取开始上网时间
  starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])
  # 将每一条记录放到onlinetimes中，并将其索引作为value赋给对应mac地址
  if mac not in mac2id:
    mac2id[mac] = len(onlinetimes)
    onlinetimes.append((starttime,onlinetime))
  else:
    onlinetimes[mac2id[mac]] = [(starttime,onlinetime)]
# 固定列数为2列，行数由系统自行确定
real_X = np.array(onlinetimes).reshape((-1,2))

# 上网时间聚类，创建DBSCAN算法实例，并进行训练，获得标签
# 上网时间数列
X = real_X[:,0:1]
# 调用DBSCAN方法进行训练
db = skc.DBSCAN(eps=0.01,min_samples=20).fit(X)
# labels为每个数据的簇标签
labels = db.labels_
print('Labels:')
print(labels)
# 计算标签为-1，即噪声数据的比例
ratio = len(labels[labels[:] == -1]) / len(labels)
print('Noise ratio:',format(ratio, '.2%'))
# 打印簇的个数并打印，评价聚类效果
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
print('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(X, labels))
# 打印各簇标号以及各簇内数据
for i in range(n_clusters_):
  print('Cluster ',i,':')
  print(list(X[labels == i].flatten()))
# 画直方图，分析实验结果
plt.hist(X,24)
plt.show()

# 分割线
print(' ')
print('------------------------------ 分割线 ------------------------------')
print(' ')

# 上网时长聚类，创建DBSCAN算法实例，并进行训练，获得标签
# 上网时长数列（由于原数据不适合聚类分析，故取对数运算）
X = np.log(1+real_X[:,1:])
# 调用DBSCAN方法进行训练
db = skc.DBSCAN(eps=0.14,min_samples=10).fit(X)
# labels为每个数据的簇标签
labels = db.labels_
print('Labels:')
print(labels)
# 计算标签为-1，即噪声数据的比例
ratio = len(labels[labels[:]==-1]) /len(labels)
print('Noise ratio:',format(ratio, '.2%'))
# 打印簇的个数并打印，评价聚类效果
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
print('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(X, labels))
# 统计每一个簇内的样本个数，均值，标准差
for i in range(n_clusters_):
  print('Cluster ',i,':')
  # 个数
  count = len(X[labels==i])
  print('\t number of sample: ',count)
  # 均值
  mean = np.mean(real_X[labels==i][:,1])
  print('\t mean of sample  : ',format(mean,'.1f'))
  # 标准差
  std = np.std(real_X[labels==i][:,1])
  print('\t std of sample   :',format(std,'.1f'))
