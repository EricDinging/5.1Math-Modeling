# 导入numpy库用于矩阵变换
import numpy as np
from numpy import shape
# 导入math库
import math
# 导入matplotlib库用于画图
import matplotlib.pyplot as plt
# 导入copy库用于实现deepcopy
import copy

# 数据处理
"""
输入：原始数据文件路径名
输出：含所有样本的list，每个元素为一个样本（含所有属性的list）
"""
def loadDataSet(fileName):
    all_data = []
    fr = open(fileName)
    for line in fr.readlines():
        oneLine = line.strip().split(',')
        all_data.append([float(oneLine[0]), float(oneLine[1])])
    return all_data

# 初始化输入层与竞争层神经元的连接权值矩阵
"""
输入：竞争层矩阵的行数n，竞争层矩阵的列数m，输入层每个样本的神经元数（属性数）d
输出：n*m*d的权值矩阵
"""
def initCompetition(n, m, d):
    array = np.random.random(size=n*m*d)
    com_weight = array.reshape(n,m,d)
    return com_weight

# 计算向量的二范数
def cal2NF(X):
    res = 0
    for x in X:
        res += x*x
    return res ** 0.5

# 对数据集进行归一化处理
def normalize(dataSet):
    for data in dataSet:
        two_NF = cal2NF(data)
        for i in range(len(data)):
            data[i] = data[i] / two_NF
    return dataSet

# 对权值矩阵进行归一化处理
def normalize_weight(com_weight):
    for x in com_weight:
        for data in x:
            two_NF = cal2NF(data)
            for i in range(len(data)):
                data[i] = data[i] / two_NF
    return com_weight

# 获得获胜神经元的索引值（利用方法一：样本和权值向量的点积）
def getWinner(data, norm_weight):
    max_sim = 0
    n,m,d = np.shape(norm_weight)
    mark_n = 0
    mark_m = 0
    for i in range(n):
        for j in range(m):
            result = sum(data * norm_weight[i][j])
            if result > max_sim:
                max_sim = result
                mark_n = i
                mark_m = j
    return (mark_n, mark_m)

# 得到拓扑邻域N中的所有神经元（计算距获胜神经元与每个神经元之间的距离，小于拓扑邻域N则是）
def getNeibor(N_neibor, com_weight):
    res = []
    n,m,_ = shape(com_weight)
    for i in range(n):
        for j in range(m):
            N_float = ((i-n)**2 + (j-m)**2) ** 0.5
            N = int(N_float)
            if N <= N_neibor:
                res.append((i,j,N))
    return res

# 学习率函数
def eta(t,N):
    return (0.3/(t+1)) * (math.e ** -N)

# 画图方法（C为聚类之后每个样本的label名称组成的list，用于分类画图）
def draw(C, dataSet):
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm', 'd']
    count = 0
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        for j in range(len(datas)):
            X.append(dataSet[datas[j]][0])
            Y.append(dataSet[datas[j]][1])
        plt.scatter(X, Y, marker = 'o', color = color[count % len(color)], label=i)
        count += 1
    plt.legend(loc='upper right')
    plt.show()
        

# SOM算法的实现（T为最大迭代次数，N_neibor是初始近邻数）
def do_som(dataSet, com_weight, T, N_neibor):
    for t in range(T-1):
        com_weight = normalize_weight(com_weight)
        for data in dataSet:
            n, m = getWinner(data, com_weight)
            neibor = getNeibor(N_neibor, com_weight)
            for x in neibor:
                j_n = x[0]; j_m = x[1]; N = x[2]
                com_weight[j_n][j_m] = com_weight[j_n][j_m] + eta(t,N)*(data - com_weight[j_n][j_m])
            N_neibor = N_neibor + 1 - (t + 1) / 200
    res = {}
    N, M, _ = shape(com_weight)
    for i in range(len(dataSet)):
        n, m = getWinner(dataSet[i], com_weight)
        key = n*M + m
        if key in res.keys():
            res[key].append(i)
        else:
            res[key] = []
            res[key].append(i)
    return res

# SOM算法主函数
def SOM(dataSet, com_n, com_m, T, N_neibor):
    old_dataSet = copy.deepcopy(dataSet)
    dataSet = normalize(dataSet)
    com_weight = initCompetition(com_n, com_m, shape(dataSet)[1])
    C_res = do_som(dataSet, com_weight, T, N_neibor)
    draw(C_res, old_dataSet)
    draw(C_res, dataSet)
    print(old_dataSet)
    print(dataSet)

fileName = '/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-聚类-SOM聚类【hxy】/dataset.txt'
dataSet = loadDataSet(fileName)
# n=2为竞争层神经元的行数，m=2为竞争层神经元的列数，T=4为最大迭代次数，N_neibor=2为初始邻域
# 可改变参数做敏感性分析
# 一般n=m=math.ceil(np.sqrt(5 * np.sqrt(样本个数)))比较合适
# 迭代次数一般可选200次
SOM(dataSet, 2, 2, 4, 2)
