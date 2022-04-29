[TOC]

### 模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】

#### 1. 模型名称

指数平滑法（Exponential Smoothing）

#### 2. 模型评价

##### 2.1 模型适用

- 一次指数平滑法：没有趋势和季节性的序列
- 二次指数平滑法：有趋势但没有季节性的序列
- 三次指数平滑法：有趋势也有季节性的序列
- 中短期经济发展趋势预测

##### 2.2 模型局限

- 不适合长期预测

#### 3. 基本算法

##### 3.1 一次指数平滑预测

当时间序列无明显的趋势变化，可用一次指数平滑预测。其预测公式为：
$$
y_{t+1}' = a y_t + (1-a) y_t'
$$
式中，$y_{t+1}'$ 为 *t* + 1 期的预测值，即本期（*t* 期）的平滑值$S_t$； $y_t$为 *t* 期的实际值； $y_t'$为 *t* 期的预测值，即上期的平滑值$S_{t − 1}$ 。

同时，称 α 为**记忆衰减因子**更合适——因为 α 的值越大，模型对历史数据“遗忘”的就越快。

一次指数平滑所得的计算结果可以在数据集及范围之外进行扩展，因此也就可以用来进行预测。预测方式为：
$$
x_{i+h} = s_i
$$
$s_i$是最后一个已经算出来的值，$h=1$代表预测的下一个值。

##### 3.2 二次指数平滑预测

![二次](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/二次.png)

- ![img](https://img-blog.csdnimg.cn/20200818223359798.png)——第*t*周期的二次指数平滑值
- ![img](https://img-blog.csdnimg.cn/20200818223420371.png)——第*t*周期的一次指数平滑值
- ![img](https://img-blog.csdnimg.cn/20200818223438584.png)——第*t*-1周期的二次指数平滑值
- $\alpha$——加权系数（也称为平滑系数)

##### 3.3 三次指数平滑预测

若时间序列的变动呈现出二次曲线趋势，则需要采用**三次指数平滑法**进行预测。三次指数平滑是在二次指数平滑的基础上再进行一次平滑，其计算公式为：

![三次1](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/三次1.png)

三次指数平滑法的预测模型为：

![三次2](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/三次2.png)

#### 4. 实例

##### 4.1 一次指数平滑预测

###### 4.1.1 问题描述

已知某种产品最近15个月的销售量如下表所示，用一次指数平滑值预测下个月的销售量$y_{16}$。

| 时间序号t   | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 销售量$y_t$ | 10   | 15   | 8    | 20   | 10   | 16   | 18   | 20   | 22   | 24   | 20   | 26   | 27   | 29   | 29   |

###### 4.1.2 数学解法

为了分析加权系数*a*的不同取值的特点，分别取$a=0.1$，$a=0.3$，$a=0.5$计算一次指数平滑值，并设初始值为最早的三个数据的平均值，以$a = 0.5$的一次指数平滑值计算为例，有
$$
S_0^{(1)} = 10.0 \\
S_1^{(1)} = \alpha y_1 + (1-\alpha) S_0^{(1)} = 0.5 \times 10+0.5\times 10.0 = 10.0 \\
S_2^{(1)} = \alpha y_2 + (1-\alpha) S_1^{(1)} = 0.5 \times 15+0.5\times 10.0 = 12.5
$$

###### 4.1.3 代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def alpha_analysis(data,itype=2):
    '''
    判断误差最小的平滑系数
    :param data:   原始序列：list
    :param itype:  平滑类型：1,2,3
    :return:       返回平均绝对误差最小的平滑系数和最小平均绝对误差
    '''
    alpha_all = [0.01 * i for i in range(1,100)]  #只需要0.1-0.9修改为alpha_triple = [0.1 * i for i in range(1,10)]
    best_alpha = 0
    min_MAE = float('Inf') #  无穷大
    if itype == 2:
        for i in range(len(alpha_all)):
            alpha = alpha_all[i]
            a_double,b_double,F_double = exponential_smoothing_2(alpha, data)
            AE_double, MAE_double, RE_double, MRE_double = model_error_analysis(F_double, data)
            if MAE_double <= min_MAE:
                min_MAE = MAE_double
                best_alpha = alpha
            else:
                pass
    elif itype == 3:
        for i in range(len(alpha_all)):
            alpha = alpha_all[i]
            a_triple, b_triple, c_triple, F_triple = exponential_smoothing_3(alpha, data)
            AE_triple, MAE_triple, RE_triple, MRE_triple = model_error_analysis(F_triple, data)
            if MAE_triple <= min_MAE:
                min_MAE = MAE_triple
                best_alpha = alpha
            else:
                pass
    else:
        for i in range(len(alpha_all)):
            alpha = alpha_all[i]
            F_single = exponential_smoothing_1(alpha, data)
            AE_single, MAE_single, RE_single, MRE_single = model_error_analysis(F_single, data)
            if MAE_single <= min_MAE:
                min_MAE = MAE_single
                best_alpha = alpha
            else:
                pass
    
    return best_alpha, min_MAE

def model_error_analysis(F, data):
    '''
    误差分析
    :param F:     预测数列：list
    :param data:  原始序列：list
    :return:      返回各期绝对误差，相对误差：list，返回平均绝对误差和平均相对误差
    '''
    AE = [0 for i in range(len(data)-1)]
    RE = []
    AE_num = 0
    RE_num = 0
    for i in range(1,len(data)):
        _AE = abs(F[i-1] - data[i])
        _RE = _AE / data[i]
        AE_num += _AE
        RE_num += _RE
        AE[i-1] = _AE
        RE.append('{:.2f}%'.format(_RE*100))
    MAE = AE_num / (len(data)-1)
    MRE = '{:.2f}%'.format(RE_num *100 / (len(data)-1))
    return AE, MAE, RE, MRE

def exponential_smoothing_1(alpha, data):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param data:   数据序列：list
    :return:       返回一次指数平滑值：list
    '''
    s_single=[]
    s_single.append((data[0]+data[1]+data[2])/3)
    for i in range(1, len(data)):
        s_single.append(alpha * data[i-1] + (1 - alpha) * s_single[i-1])
    return s_single

t = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
data = [10,15,8,20,10,16,18,20,22,24,20,26,27,29,29]
alpha_analysis(data,itype=1)
alpha = 0.5
s = exponential_smoothing_1(alpha, data)
print(s)
```

结果：($\alpha =0.5$)

![result1](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/result1.png)

##### 4.2 二次指数预测模型

###### 4.2.1 问题描述

某地1983年至1993年财政入的资料如下，试用指数平滑法求解趋势直线方程并预测1996年的财政收入

###### 4.2.2 数学解法

![二次实例1](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/二次实例1.png)

![二次实例2](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/二次实例2.png)

##### 4.3 三次指数平滑预测

###### 4.3.1 问题描述

我国某种耐用消费品1996年至2006年的销售量如表所示，试预测2007、2008年的销售量。

###### 4.3.2 数学解法

 三次指数平滑的计算表：  

![三次实例1](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/三次实例1.png)

通过实际数据序列呈非线性递增趋势，采用三次指数平滑预测方法。确定指数平滑的初始值和权系数（平滑系数）a。设一次、二次指数平滑的初始值为最早三个数据的平均值，即

![三次实例0](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/三次实例0.png)

取
$$
S_0^{(3)}  =244.5
$$
实际数据序列的倾向性变动较明显，权系数（平滑系数）*a* 不宜取太小，故取*a*= 0.3。 

根据指数平滑值计算公式依次计算一次、二次、三次指数平滑值：

![三次实例2](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/三次实例2.png)

计算非线性预测模型的系数：

![三次实例3](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/三次实例3.png)

建立非线性预测模型得：

![三次实例4](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/三次实例4.jpg)

预测2007年和2008年的产品销售量。2007年，其预测超前周期为*T*= 1；2008年，其预测超前周期为*T*= 2。代入模型，得

![三次实例5](/Users/xinyuanhe/Desktop/【正式】模型-预测主题-连续型预测时间序列模型-指数平滑法【hxy】/三次实例5.jpg)

于是得到2007年的产品销售量的预测值为809万台，2008年的产品销售量的预测值为920万台。预测人员可以根据市场需求因素的变动情况，对上述预测结果进行评价和修正。

#### 5. 参考资料

1. [预测算法——指数平滑法](https://blog.csdn.net/meng_shangjy/article/details/79972512?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.pc_relevant_antiscanv2&spm=1001.2101.3001.4242.1&utm_relevant_index=3)

2. [参考代码](https://github.com/ishelo/Logistics-Demand-Forecasting-By-Python)