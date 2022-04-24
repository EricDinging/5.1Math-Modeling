# 导入sklearn相关包
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
# 导入matplotlib
import matplotlib.pyplot as plt
# 导入pandas
import pandas as pd

# 加载iris数据
iris = datasets.load_iris()
irisdata = iris.data

# 建立AGNES模型
# linkage是一个字符串，用于指定链接算法，ward表示采用单链接，complete表示全链接，average表示均链接
# n_clusters指定分类簇的数量
clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)

# 输入iris数据进行训练
res = clustering.fit(irisdata)

# 打印各个簇的样本数目
print('各个簇的样本数目：')
print(pd.Series(clustering.labels_).value_counts())
# 打印聚类结果
print('聚类结果：')
print(confusion_matrix(iris.target,clustering.labels_))

# 可视化
plt.figure()
# labels_为0的点
d0 = irisdata[clustering.labels_==0]
plt.plot(d0[:,0],d0[:,1],'r.')
# labels_为1的点
d1 = irisdata[clustering.labels_==1]
plt.plot(d1[:,0],d1[:,1],'go')
# labels_为2的点
d2 = irisdata[clustering.labels_==2]
plt.plot(d2[:,0],d2[:,1],'b*')
# 设置xlabel和ylabel
plt.xlabel('Sepal.Length')
plt.ylabel('Sepal.Width')
# 设置title
plt.title('AGNES Clustering')
# 显示
plt.show()
