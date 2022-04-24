[TOC]

### 模型-经济管理-市场与资产模型-巴斯扩散模型【hxy】

#### 1. 模型名称

巴斯扩散模型（Bass Diffusion Model）

#### 2. 基本内容

##### 2.1 假设

- 市场潜力随时间的推移保持不变
- 一种创新的扩散独立于其他创新
- 产品性能随时间推移保持不变
- 社会系统的地域界限不随扩散过程而改变
- 扩散只有两阶段过程：采用和不采用
- 一种创新的扩散不受市场营销策略的影响
- 不存在供给约束
- 采用者是无差异的、同质的

##### 2.2 模型推导

**参数说明**：
$$
m - 市场总潜力(最终采用者总数)\\
p - 创新参数\\
q- 模仿参数\\
f(t) - 在时间t时的采用者数量占总的潜在采用者数量比例的概率密度函数\\
F(t) - 在时间t的采用者的累计比例\\
n(t) - 在时间t的采用者的数量\\
N(t) - 到时间t时累积采用者数量\\
$$
**模型建立**：
$$
\frac{dN(t)}{dt} = p[m-N(t)] + q\frac{N(t)}{m}[m-N(t)], \;
N(t) = mF(t) \\
其中，p[m-N(t)]指因为外部影响而购买新产品的采用人数，\\q\frac{N(t)}{m}[m-N(t)]指受先前购买者影响而购买的采用人数
$$
**初始条件**：
$$
当t=0时，n(0)=pm\quad \\即在创新扩散开始时，有pm个采用者，也可以理解为新产品引入市场前的试销或赠送的样品
$$
**结果**：
$$
N(t) = m \frac{1 - e^{-(p+q)t}}{1+(\frac{q}{p}) e^{-(p+q)t}},\quad n(t) = m \frac{p(p+q)^2 e^{-(p+q)t}}{[p+q e^{-(p+q)t}]^2}
$$
 **图示**：

- 如果$q>p$，则采纳曲线由最高点，即此产品的扩散是成功的如果
- 如果$q\leq p$，则增长曲线没有极值点，随时间呈指数衰减状态，说明此产品的市场扩散失败

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-经济管理-市场与资产模型-巴斯扩散模型【hxy】/20200411185454161.png" alt="20200411185454161" style="zoom:80%;" />

##### 2.3 应用和局限

**适用范围**：

- 耐用消费品的分析预测，既适用于新产品，也适用于已进入市场的产品
- 简洁明了，适用于初次评估
- 变形模型，可以适用于一些特殊情况

**局限**：

- 巴斯模型给出的是购买者数量，而不是企业的产品销售量，但是销售量可以根据顾客使用频率间接估计
- 虽然巴斯模型在理论上比较完善，但是只适用于已经在市场中存在一定时期的新产品的市场预测，而往往新产品上市的时候，其治疗和性能对顾客来讲相当陌生，企业无法对巴斯模型中的创新系数和模仿系数作出可靠的估计，此时就需要对巴斯扩散模型做出一定的补充

#### 3. 代码实现

 [bass.py](bass.py) 

代码：

```python
# 最小二乘法
from math import e  # 引入自然数e
import numpy as np  # 科学计算库
import matplotlib.pyplot as plt  # 绘图库
from scipy.optimize import leastsq  # 引入最小二乘法算法

# 样本数据(Xi,Yi)，需要转换成数组(列表)形式
ti = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
yi = np.array([8, 11, 15, 19, 22, 23, 22, 19, 15, 11])


# 需要拟合的函数func :指定函数的形状，即n(t)的计算公式
def func(params, t):
    m, p, q = params
    fz = (p * (p + q) ** 2) * e ** (-(p + q) * t)  # 分子的计算
    fm = (p + q * e ** (-(p + q) * t)) ** 2  # 分母的计算
    nt = m * fz / fm  # nt值
    return nt


# 误差函数函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
# 一般第一个参数是需要求的参数组，另外两个是x,y
def error(params, t, y):
    return func(params, t) - y


# k,b的初始值，可以任意设定, 一般需要根据具体场景确定一个初始值
p0 = [100, 0.3, 0.3]

# 把error函数中除了p0以外的参数打包到args中(使用要求)
params = leastsq(error, p0, args=(ti, yi))
params = params[0]

# 读取结果
m, p, q = params
print('m=', m)
print('p=', p)
print('q=', q)

# 有了参数后，就是计算不同t情况下的拟合值
y_hat = []
for t in ti:
    y = func(params, t)
    y_hat.append(y)

# 接下来我们绘制实际曲线和拟合曲线
# 由于模拟数据实在太好，两条曲线几乎重合了
fig = plt.figure()
plt.plot(yi, color='r', label='true')
plt.plot(y_hat, color='b', label='predict')
plt.title('BASS model')
plt.legend()
plt.show()
```

结果：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-经济管理-市场与资产模型-巴斯扩散模型【hxy】/result.png" alt="result" style="zoom:67%;" />

![Screen Shot 2022-02-12 at 10.12.11 PM](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-经济管理-市场与资产模型-巴斯扩散模型【hxy】/Screen Shot 2022-02-12 at 10.12.11 PM.png)

#### 4. 阅读材料

1.  [品牌资产理论在中国的发展阶...特征——基于扩散模型的研究_顾雷雷.pdf](品牌资产理论在中国的发展阶...特征——基于扩散模型的研究_顾雷雷.pdf) 
2.  [绿色供应链管理创新扩散趋势...究——基于中国省际面板数据_刘娜.pdf](绿色供应链管理创新扩散趋势...究——基于中国省际面板数据_刘娜.pdf) 
3.  [基于Multi-Agent的虚假舆情传播仿真_孙雷霆.pdf](基于Multi-Agent的虚假舆情传播仿真_孙雷霆.pdf) 

#### 5. 参考资料

1. [巴斯扩散模型](https://blog.csdn.net/qq_41103204/article/details/105437287)

2. [寒假第十五次培训-经管类模型概览-巴斯扩散模型](https://vshare.sjtu.edu.cn/play/cd8ea54e5f1b42cf7229ef9202c8c9df)