# 导入sklearn相关包
import numpy as np
from sklearn.cluster import KMeans

# 自定义加载数据函数
def loadData(filePath):
  # 以读的兼容模式打开文件
  fr = open(filePath, 'r+')
  # 读取每一行
  lines = fr.readlines()
  # retData用来存储城市各项消费信息
  retData = []
  # retCityName用来存储城市名称
  retCityName = []
  # 将每一行城市信息分别存到retData和retCityName中
  for line in lines:
    items = line.strip().split(",")
    retCityName.append(items[0])
    retData.append([float(items[i]) for i in range(1,len(items))])
  # 返回retData和retCityName
  return retData, retCityName

# 加载数据，创建K-means算法实例，并进行训练，获得标签
if __name__ == '__main__':
  # 用自定义loadData方法读取数据
  data, cityName = loadData('/Users/xinyuanhe/Desktop/city.txt')
  # 创建实例
  # n_clusters用于指定聚类中心的个数，init初始聚类中心的初始化方法，max_iter最大迭代次数
  # init默认k-means++，max_iter默认300
  km = KMeans(n_clusters=3)
  # 调用fit_predict进行计算：fit算簇中心，predict指定x中每个点所属于的簇的位置
  label = km.fit_predict(data)
  # 计算不同簇的平均花费
  # np.sum(axis=1)计算每一行向量的和
  # km.cluster_centers_聚类中心
  expenses = np.sum(km.cluster_centers_,axis=1)
  # 创建簇
  CityCluster = [[],[],[]]
  # 将每个样本分到不同簇中
  for i in range(len(cityName)):
    CityCluster[label[i]].append(cityName[i])
  # 打印每个簇的复杂度和簇中样本
  for i in range(len(CityCluster)):
    print("Expenses:%.2f" % expenses[i])
    print(CityCluster[i])
