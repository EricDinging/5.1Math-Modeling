[TOC]

### 模型-概率统计-连续型概率分布-均匀分布

#### 1. 模型名称

均匀分布（Uniform distribution）

连续型均匀分布（continuous uniform distribution）

#### 2. 模型表述

##### 2.1 数学化表述

如果连续型随机变量X具有如下的概率密度函数)，
$$
\begin{aligned}

\text { f(x)= } &\left\{\begin{array}{l}
\frac{1}{b-a} &for\space  a\le x \le b\\
0 &elsewhere\

\end{array}\right.
\end{aligned}
$$
则称X服从 [a,b]上的均匀分布（uniform distribution），记作
$$
X∼U[a,b]
$$

##### 2.2 累计分布函数

$$
\begin{aligned}

\text { F(x) =} &\left\{\begin{array}{l}
0 &for\space   x <a\\
\frac{x-a}{b-a}  &for\space  a\le x \le b\\
1&  for\space x\ge b

\end{array}\right.
\end{aligned}
$$

##### 2.3 期望和中值

是指连续型均匀分布函数的期望值和中值等于区间[a,b]上的中间点。
$$
E[X]=\frac{a+b}{2}
$$

##### 2.4 方差

$$
VAR[X]=\frac{(b-a)^2}{12}
$$

##### 2.5 下属意义的等可能性

均匀分布具有下属意义的等可能性。若$X∼U[a,b]$，则X落在[a,b]内任一子区间[c,d]上的概率
$$
P(c\le x\le d)=F(d)-F(c)=\int_c^d\frac{1}{b-a}dx=\frac{d-c}{b-a}
$$
只与区间[c,d]的长度有关，而与他的位置无关。
