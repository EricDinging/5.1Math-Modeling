[TOC] 

### 模型-机器学习-聚类-SOM聚类

#### 1. 模型名称

自组织特征图（Self-Organizing Feature Map，SOM）

#### 2. 模型评价

##### 2.1 缺点

1. 网络结构是固定的，不能动态改变

2. 网络训练时,有些神经元始终不能获胜，成为“死神经元”

3. SOM 网络在没有经过完整的重新学习之前，不能加入新的

4. 当输入数据较少时，训练的结果通常依赖于样本的输入顺序
5. 网络连接权的初始状态、算法中的参数选择对网络的收敛性能有较大影响

##### 2.2 优点

1. 简明性
2. 实用性

#### 3. 基本算法

1. 输入**输入层**：假设一个输入样本$X = [x_1,x_2,x_3,...,x_n]$，则输入层神经元的个数为$n$个

2. 用较小的随机值初始化**输出层（竞争层）**：通常输出层的神经元以矩阵方式排列在二维空间中，$m$个神经元有$m$个**权值向量**$w_i = [w_{i1}, w_{i2}, ... , w_{in}], \; i = 1,2,..,m$

3. 对输入向量做**归一化**：$||X||$为输入的样本向量的欧几里得范数
   $$
   X' = \frac{X}{||X||}
   $$

4. 对权值向量做**归一化**：$||w_i||$为权值向量的欧几里得范数
   $$
   w_i'= \frac{w_i}{||w_i||}
   $$

5. 得到**获胜神经元**

   方法一：样本$X$和每个竞争层的神经元的权值向量$w_i$**点积**，值最大的为**获胜神经元**

   方法二：计算样本$X$和每个竞争层的神经元的权值向量的**欧几里得距离**，距离最小的为**获胜神经元**

6. 得到获胜神经元**拓扑邻域**$N$内的神经元

7. 对获胜神经元**拓扑邻域$N$**内的每个神经元进行**权值更新**和**归一化**，并更新**学习速率**$\eta$和**拓扑邻域**$N$
   $$
   w_i(t+1) = w_i(t) + \eta(t,d) * (X - w_i(t)) \\
   \eta(t,d):\eta为学习速率，t为训练时间，d为该神经元与获胜神经元之间的拓扑距离 \\
   \eta(t,d) = \eta(t)e^{-d}, \quad \eta(t)一般取迭代次数的倒数
   $$

8. 判断是否收敛：如果**学习率**$\eta<\eta_{min}$或者**迭代次数**$t>T$，则结束算法

#### 4. 实例

##### 4.1 数据介绍

来源于西瓜书，一共有30个西瓜，每个西瓜有2个不同属性

```
0.697,0.46
0.774,0.376
0.634,0.264
0.608,0.318
0.556,0.215
0.403,0.237
0.481,0.149
0.437,0.211
0.666,0.091
0.243,0.267
0.245,0.057
0.343,0.099
0.639,0.161
0.657,0.198
0.36,0.37
0.593,0.042
0.719,0.103
0.359,0.188
0.339,0.241
0.282,0.257
0.748,0.232
0.714,0.346
0.483,0.312
0.478,0.437
0.525,0.369
0.751,0.489
0.532,0.472
0.473,0.376
0.725,0.445
0.446,0.459
```

##### 4.2 实验目的

根据西瓜的2个属性，对30个西瓜进行分类

##### 4.3 代码实现

###### 4.3.1 Python自写代码

 [SOM.py](SOM.py) 

```python
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
```

原始数据聚类图：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-聚类-SOM聚类【hxy】/原始数据聚类图.png" alt="原始数据聚类图" style="zoom: 33%;" />

归一化后数据聚类图：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-聚类-SOM聚类【hxy】/归一化后数据聚类图.png" alt="归一化后数据聚类图" style="zoom:33%;" />

原始数据和归一化后数据：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-聚类-SOM聚类【hxy】/原始数据与归一化数据list.png" alt="原始数据与归一化数据list" style="zoom: 50%;" />

###### 4.3.2 github-minisom库的调用

 [minisom.py](minisom/minisom.py) 

 [test_minisom.py](test_minisom.py) 

```python
# install
pip3 install minisom
```

```python
# use
# import minisom
from minisom import MiniSom
import matplotlib.pyplot as plt
# input data
data = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]
# set parameters
som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
# train
som.train(data, 100) # trains the SOM with 100 iterations
# draw U-Matrix
heatmap = som.distance_map()  #生成U-Matrix
plt.imshow(heatmap, cmap='bone_r')      #miniSom案例中用的pcolor函数,需要调整坐标
plt.colorbar()
plt.show()
# print
print(som.get_weights())
```

画U-Matrix图（用距离显示关联度）：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-聚类-SOM聚类【hxy】/minisom_plot.png" alt="minisom_plot" style="zoom: 33%;" />

输出权值向量：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-聚类-SOM聚类【hxy】/minisom_weight.png" alt="minisom_weight" style="zoom: 33%;" />

#### 5. 参考资料

1. [SOM-github源码整理](https://zhuanlan.zhihu.com/p/73534694)
2. [github-minisom源码](https://github.com/JustGlowing/minisom)
3. [SOM-可视化](https://zhuanlan.zhihu.com/p/73930638)

4. [数模官网-SOM聚类](https://anl.sjtu.edu.cn/mcm/docs/1%E6%A8%A1%E5%9E%8B/3%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2%E8%81%9A%E7%B1%BB/SOM%E8%81%9A%E7%B1%BB/doc)

5. [机器学习实战——python实现SOM神经网络聚类算法](https://blog.csdn.net/chenge_j/article/details/72537568?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.pc_relevant_default&spm=1001.2101.3001.4242.1&utm_relevant_index=3)（偏代码实现）

6. [机器学习实战——python实现DBSCAN密度聚类](https://blog.csdn.net/chenge_j/article/details/72357471?spm=1001.2014.3001.5502)（用于2的数据处理参考）

7. [SOM算法](https://blog.csdn.net/weixin_38347387/article/details/80342662)（偏数学理论）