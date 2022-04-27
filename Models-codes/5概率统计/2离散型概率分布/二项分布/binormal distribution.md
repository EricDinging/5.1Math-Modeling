---
id: doc
title: 二项分布   
---           
## 1. 模型概览

### 1.1 学科分属

> 概率统计，离散型随机变量。需要基本的排列组合知识。

### 1.2 历史发展

> 待添加

## 2. 模型介绍

### 2.1  具体模型介绍

> 定义一个随机过程的成功概率为 $p$, $0<p<1$。重复这个随机过程$n$次，成功$x$次的概率为
$$f_X(x)=\tbinom{n}{x}p^x(1-p)^{n-x}$$
其中$X$被称为
>binormal random variable with parameters $n$ and $p$

其积累函数(cumulative distribution function)为
$$F_X(x)=\sum_{y\leqslant x}f_X(y)=\sum^{\lfloor x \rfloor}_{y=0}\tbinom{n}{y}p^y(1-p)^{n-y}$$
>期望为
$$E[X]=np$$
>方差为
$$Var[X]=np(1-p)$$

### 2.2  模型变种
>对二项分布的估计
##### 泊松分布
对于非常大的$n$，
$$\tbinom{n}{x}p^x(1-p)^{n-x}   \longrightarrow \frac{k^m}{m!}e^{-k}$$
其中$k=n\cdot p$
#### 正态分布
对于$np>5, p\leqslant1/2$ 或者$n(1-p)>5, p>1/2$
$$P[X\leqslant y]=\sum^{y}_{x=0}\tbinom{n}{x}p^x(1-p)^{n-x}\approx \Phi\large(\frac{y+1/2-np}{\sqrt{np(1-p)}})$$

### 2.3  ……

> 可能还有更多其他内容

#### 2.3.1 更多级列表

##### 2.3.1.1 超多级列表（尽量不要太多级）

## 3. 模型应用

### 3.1 常见应用场景

>计算一个只有两种可能的随机变量重复n次后得到想要结果的概率。

### 3.2 数学建模竞赛应用

> 阐述使用该模型的美赛、国赛等建模试题。

### 3.3 ……

> 可能还有其他领域的应用与拓展

## 4.软件/程序介绍

### 4.1 模拟介绍

> Mathematica


```Mathematica
PDF[BinomialDistribution[5, .2], 1];
CDF[BinomialDistribution[10, .5], 3];
CDF[BinomialDistribution[n, p], x] /. {n -> 10, p -> 0.5, x -> 3}; 
InverseCDF[BinomialDistribution[10, .5], 11/64];
Variance[BinomialDistribution[10, .2]];
MomentGeneratingFunction[BinomialDistribution[10, .2], x];
```

### 4.2 其他特殊软件实现

> 如 Lingo、CPlex

### 4.3 ……

## 修改记录

- 2021-08-
- 2021-08-10，郑鸿晓更新模板
- 2021-08-04，张嘉乐建立模板

> 请在论文最后加入参考文献，在正文中详细引用
> 
> 注：若在正文中不引用，则不会显示该注脚！务必在正文中引用

这是正文[^1]

[^1]: 全国信息与文献标准委员会第6分委员会. GB/T 7714—2005 文后参考文献著录规则[S].北京：中国标准出版社，2005.