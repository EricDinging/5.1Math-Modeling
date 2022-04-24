[TOC] 

### 模型-机器学习-回归-线性回归【hxy】

#### 1. 模型名称

线性回归（Linear Regression，LP）

#### 2. 适用范围

1. **线性关系分析**：给定一个变量$Y$和一些变量$X_1,...,X_p$，线性回归可以量化$Y$与$X_i$之间相关性的强度，评估出与$y$不相关的$X_i$，并识别出哪些$X_i$的子集包含了关于$Y$的冗余信息

2. **预测**或**映射**：对于一个在模型范围内新增的x值，没有给定其y值的情况下，可以用模型预测出一个y值
3. **反预测**：给定一个变量$Y$，求出$x$的范围

4. **控制**：给定一个变量$Y$的区间，求出$x$的范围

#### 3. 形式

1. **简单回归**：一个因变量$Y$，一个自变量$X$
2. **多元回归**：一个因变量$Y$，多个自变量$X_i$

#### 4. 求解过程

##### 4.1 概念

利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法

##### 4.2 步骤

###### 4.2.1 简单回归

1. 解**回归方程**（最小二乘法）

   a) 设回归方程结构
   $$
   y = \beta_0 + \beta_1 x + \varepsilon, \; E(\varepsilon) = 0, \; D(\varepsilon) = \sigma^2
   $$
   b) 解平均值$\overline x$和$\overline y$（n为样本数，t为每一个样本）
   $$
   \overline x = \frac{1}{n}\sum_{t = 1}^n x_t, \; \overline y = \frac{1}{n}\sum_{t = 1}^n y_t
   $$
   c) 解方差$L_{xy}$，$L_{xx}$和$L_{yy}$
   $$
   L_{xy} = \sum_{t = 1}^n (x_t - \overline x)(y_t - \overline y), \; 
   L_{xx} = \sum_{t = 1}^n (x_t - \overline x)^2, \;
   L_{yy} = \sum_{t = 1}^n (y_t - \overline y)^2
   $$
   d) 解$\widehat \beta_1$和$\widehat \beta_0$
   $$
   \widehat \beta_1 = \frac{L_{xy}}{L_{xx}}, \; \widehat \beta_0 = \overline y - \overline x \;\widehat \beta_1
   $$
   e) 得到**回归方程**
   $$
   \widehat y = \widehat \beta_0 + \widehat \beta_1 x
   $$

2. **检验线性关系显著性**

   a) 解**回归平方和**$U_R$
   $$
   U_R = {\widehat \beta_1}^2 L_{xx}
   $$
   b) 解**残差平方和**$Q_e$
   $$
   Q_e = L_{yy} - U_R
   $$
   c) 解$F$
   $$
   F = \frac{U_R }{Q_e / (n-2)}
   $$
   d) 检验**显著性**

   在**显著性水平**$\alpha = 0.05$下，查表得$F_{0.95}(1, (n - 2))$的值，若$F>F_{0.95}(1, (n-2))$，则$y$与$x$之间线性关系显著，且F越大越显著

3. 应用

   - **预测**

     a) 计算$x_0$的**预测值**$\widehat y_0$
     $$
     \widehat y_0 = \widehat \beta_0 + \widehat \beta_1 x_0
     $$
     b) 计算$\widehat \sigma_e$
     $$
     \widehat \sigma_e = \sqrt{\frac{Q_e}{n - 2}}
     $$
     c) 查表得$t_{1-\alpha/2}(n -2)$的值

     当**显著性水平**为$\alpha=0.05$时，查表得$t_{0.975}(n - 2)$的值

     d) 计算$x_0$的**预测区间**$(\widehat y_2, \widehat y_1)$
     $$
     \left\{\begin{array}{l}
     \hat{y}_{1}=\hat{\beta}_{0}+\hat{\beta}_{1}x_0+\hat{\sigma}_{e}t_{1-\alpha / 2}(n-2) \sqrt{1+\frac{1}{n}+\frac{(x_0-\bar{x})^{2}}{L_{x x}}} \\
     \hat{y}_{2}=\hat{\beta}_{0}+\hat{\beta}_{1}x_0-\hat{\sigma}_{e} t_{1-\alpha / 2}(n-2) \sqrt{1+\frac{1}{n}+\frac{(x_0-\bar{x})^{2}}{L_{x x}}}
     \end{array}\right.
     $$

   - **反预测**

     a) 计算$\widehat \sigma_e ^2$
     $$
     \widehat \sigma_e ^2 = \frac{Q_e}{n - 2}
     $$
     b) 算$d_1$和$d_2$
     $$
     d^2 [\hat{\beta_1}^2 - \frac{\widehat \sigma_e^2 F_{1-\alpha}(1,n-2)}{L_{xx}}] - 2 \hat{\beta_1}(y_0 - \overline y)d + [(y_0 - \overline y)^2 - \widehat \sigma_e^2 F_{1-\alpha}(1,n-2)(1+\frac{1}{n})] = 0
     $$
     c) 算$x_0$的$1-\alpha$**置信区间**$(d_1 + \overline x, d_2 + \overline x)$

   - **控制**

     a) 查**U值表**得$u_{1-\alpha/2}$

     当**显著性水平**$\alpha=0.05$时，$u_{0.975}=1.96$

     b) 计算$x_1$和$x_2$（$y_1$为给定区间下界，$y_2$为给定区间上界）

     注意：仅当$n$足够大且$x$与$\overline x$接近时成立
     $$
     x_1 = \frac{1}{\widehat \beta_1} (y_1 + \widehat \sigma_e u_{1 - \alpha/2} - \widehat \beta_0) \\
     x_2 = \frac{1}{\widehat \beta_1} (y_2 - \widehat \sigma_e u_{1-\alpha/2} - \widehat \beta_0)
     $$
     c) 得到**控制区间**

     当$\widehat \beta_1 > 0$时，$x$的$1-\alpha$的控制区间为$(x_1,x_2)$；

     当$\widehat \beta_1 < 0$时，$x$的$1-\alpha$的控制区间为$(x_2,x_1)$

###### 4.2.2 多元回归

1. 解**回归方程**（最小二乘法）

   a) 建立$X'$
   $$
   X' = 
   \begin{bmatrix}
   1 & 1 & ... & 1 \\
   x_{11} & x_{12} & ... & x_{1n}\\
   x_{21} & x_{22} & ... & x_{2n}\\
   ... & ... & ... & ...\\
   x_{k1} & x_{k2} & ... & x_{kn}
   \end{bmatrix}
   $$
   b) 计算$X'X$

   c) 计算$C$
   $$
   C = L^{-1} = (X'X)^{-1}
   $$
   d) 计算$X'Y$

   e) 计算$\widehat \beta$
   $$
   \widehat \beta = 
   \begin{bmatrix}
   \widehat \beta_0 \\
   \widehat \beta_1 \\
   ... \\
   \widehat \beta_k
   \end{bmatrix} 
    = C(X'Y)
   $$
   f) 得到**回归方程**
   $$
   \widehat y = \widehat \beta_0 + \widehat \beta_1 x_1 + ... + \widehat \beta_k x_k
   $$

2. **检验线性关系显著性**

   a) 检验**整体**线性关系（即检验假设$H_0: \beta_1 = \beta_2 = ... = \beta_k = 0$）
   $$
   S_总 = \sum_{t = 1}^n (y_t - \overline y)^2 = \sum_{t = 1}^n y_t^2 -n\overline y^2 \\
   U_R = \sum_{t = 1}^n (\sum_{i = 1}^k \widehat \beta_i x_{ti})^2\\
   Q_e = S_总 - U_R \\
   F = \frac{U_R /k}{Q_e/(n - k -1)}
   $$
   对**显著性水平**$\alpha=0.05$查**F分布表**得临界值$F_{1-\alpha}(k,n-k-1)$的值$F_{0.95}(k,n-k-1)$，若$F > F_{0.95}(k,n-k-1)$，则否定$H_0$，即认为$\beta_1,\beta_2,\beta_3$不全为零，数据**具有线性**

   b) 检验**回归系数**
   $$
   C = L^{-1} = [c_{ij}] \\
   F_i = \frac{U_i}{Q_e / (n - k - 1)} = \frac{\widehat \beta_i^2 / c_{ii}}{Q_e / (n - k - 1)}
   $$
   对**显著性水平**$\alpha=0.05$查**F分布表**得临界值$F_{1-\alpha}(1,n-k-1)$的值$F_{0.95}(1,n-k-1)$

   情况一：若$F_i > F_{0.95}(1,n-k-1),i = 1,2,...,k$，则原回归方程所有自变量都具有**显著性**，原回归方程即最终回归方程

   情况二：若存在$F_i < F_{0.95}(1,n-k-1),i = 1,2,...,k$，则剔除$F_i$最小值对应的自变量，建立新的回归方程，并重复第1步和第2步

3. 应用-**预测**（反预测与控制太过复杂，不讨论）

   a) 计算**预测值**$\widehat y_0$
   $$
   \widehat y_0 = \widehat \beta_0 + \widehat \beta_1 x_1 + ... + \widehat \beta_k x_k
   $$
   b) 计算$\widehat \sigma_e$
   $$
   \widehat \sigma_e = \sqrt{\frac{Q_e}{n - 2}}
   $$
   c) 查表得$t_{1-\alpha/2}(n -2)$的值

   当**显著性水平**为$\alpha=0.05$时，查表得$t_{0.975}(n - 2)$的值

   d) 计算**预测区间**$(\widehat y_2, \widehat y_1)$
   $$
   \left\{\begin{array}{l}
   \hat{y}_{1}=\hat{y}-\hat{\sigma}_{e} t_{1-\alpha / 2}(n-k-1) \sqrt{1+\sum \sum c_{i j} x_{\dot{i}} x_{j}} \\
   \hat{y}_{2}=\hat{y}+\hat{\sigma}_{e} t_{1-\alpha / 2}(n-k-1) \sqrt{1+\sum \sum c_{i j} x_{i} x_{j}}
   \end{array}\right.
   $$

##### 4.3 实例

###### 4.3.1 简单回归

以身高$x$为横坐标，腿长$y$为纵坐标，研究两者的关系

| 身高 | 143  | 145  | 146  | 147  | 149  | 150  | 153  | 154  | 155  | 156  | 157  | 158  | 159  | 160  | 162  | 164  |
| ---- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 腿长 |  88  |  85  |  88  |  91  |  92  |  93  |  93  |  95  |  96  |  98  |  97  |  96  |  98  |  99  | 100  | 102  |

1. 解**回归方程**（最小二乘法）

   a) 设回归方程结构
   $$
   y = \beta_0 + \beta_1 x + \varepsilon, \; E(\varepsilon) = 0, \; D(\varepsilon) = \sigma^2
   $$
   b) 解平均值$\overline x$和$\overline y$（n为样本数，t为每一个样本）
   $$
   \overline x = \frac{1}{16}\sum_{t = 1}^{16}x_t = 153.625, \; \overline y = \frac{1}{16}\sum_{t = 1}^{16} y_t = 94.4375
   $$
   c) 解方差$L_{xy}$，$L_{xx}$和$L_{yy}$
   $$
   L_{xy} = \sum_{t = 1}^{n} (x_t - \overline x)(y_t - \overline y) = \sum_{t = 1}^{16} (x_t - 153.625)(y_t - 94.4375) = 438.625 \\
   L_{xx} = \sum_{t = 1}^{n} (x_t - \overline x)^2  = \sum_{t = 1}^{16} (x_t - 153.625)^2 = 609.75 \\
   L_{yy} = \sum_{t = 1}^{n} (y_t - \overline y)^2 = \sum_{t = 1}^{16} (y_t - 94.4375)^2 = 339.9375
   $$
   d) 解$\widehat \beta_1$和$\widehat \beta_0$
   $$
   \widehat \beta_1 = \frac{L_{xy}}{L_{xx}} = \frac{438.625}{609.75} = 0.719 \\ \widehat \beta_0 = \overline y - \overline x \;\widehat \beta_1 = 94.4375 - 153.625 \times 0.719 = -16.07
   $$
   e) 得到**回归方程**
   $$
   \widehat y = \widehat \beta_0 + \widehat \beta_1 x = -16.07 + 0.719x
   $$

2. **检验线性关系显著性**

   a) 解**回归平方和**$U_R$
   $$
   U_R = {\widehat \beta_1}^2 L_{xx} = 0.719^2 \times 609.75 = 315.217 
   $$
   b) 解**残差平方和**$Q_e$
   $$
   Q_e = L_{yy} - U_R = 339.9375 - 315.217 = 24.7205
   $$
   c) 解$F$
   $$
   F = \frac{U_R }{Q_e / (n-2)} = \frac{315.217 \times 14}{24.7205} = 178.517
   $$
   d) 检验**显著性**

   在**显著性水平**$\alpha = 0.05$下，查表得$F_{0.95}(1, 14) = 4.6$的值，$F=178.517 >F_{0.95}(1, 14)=4.6$，则$y$与$x$之间线性关系非常显著

3. 应用

   - **预测**

     设$x_0=170$，求相应的$y_0$的预测值和预测区间（$\alpha=0.05$）

     a) 计算$x_0=170$的**预测值**$\widehat y_0$
     $$
     \widehat y_0 = \widehat \beta_0 + \widehat \beta_1 x_0 = -16.07 + 0.719 \times 170 = 106.16
     $$
     b) 计算$\widehat \sigma_e$
     $$
     \widehat \sigma_e = \sqrt{\frac{Q_e}{n - 2}} = \sqrt{\frac{24.7205}{14}} = 1.329
     $$
     c) 查表得$t_{1-\alpha/2}(n -2)$的值

     当**显著性水平**为$\alpha=0.05$时，查表得$t_{0.975}(14) = 2.1448$

     d) 计算$x_0$的**预测区间**$(\widehat y_2, \widehat y_1)$
     $$
     \hat{y}_{1}=\hat{\beta}_{0}+\hat{\beta}_{1}x_0+\hat{\sigma}_{e}t_{1-\alpha / 2}(n-2) \sqrt{1+\frac{1}{n}+\frac{(x_0-\bar{x})^{2}}{L_{x x}}} \\= -16.07 + 0.719 \times 170 + 1.329 \times 2.1448 \times \sqrt{1+\frac{1}{16}+\frac{(170-153.625)^2}{609.75}} = 109.654\\
     \hat{y}_{2}=\hat{\beta}_{0}+\hat{\beta}_{1}x_0-\hat{\sigma}_{e} t_{1-\alpha / 2}(n-2) \sqrt{1+\frac{1}{n}+\frac{(x_0-\bar{x})^{2}}{L_{x x}}} \\
     =-16.07 + 0.719 \times 170 -1.329 \times 2.1448 \times \sqrt{1+ \frac{1}{16}+ \frac{(170 - 153.625)^2}{609.75}} = 102.666
     $$
     于是$y$的预测值为$106.16$，其$0.95$的预测区间为$(102.666,109.654)$
   
   - **反预测**
   
     当$y_0=110$时，求相应的自变量$x_0$的$0.95$的置信区间
   
     a) 计算$\widehat \sigma_e ^2$
     $$
     \widehat \sigma_e ^2 = \frac{Q_e}{n - 2} = \frac{24.7205}{14} = 1.7658
     $$
     b) 算$d_1$和$d_2$
     $$
     d^2 [\hat{\beta_1}^2 - \frac{\widehat \sigma_e^2 F_{1-\alpha}(1,n-2)}{L_{xx}}] - 2 \hat{\beta_1}(y_0 - \overline y)d + [(y_0 - \overline y)^2 - \widehat \sigma_e^2 F_{1-\alpha}(1,n-2)(1+\frac{1}{n})] = 0  \\
     \Rightarrow [0.719^2 - \frac{1.7658 \times 4.6}{609.75}]d^2 -2 \times 0.719 \times(110-94.4375)d+ \\ [(110-94.4375)^2-1.7658 \times 4.6 \times (1 + \frac{1}{16})] =0 \\
     \Rightarrow 0.5043d^2 - 22.3898d + 233.6691 = 0 
     \Rightarrow d_1 = 16.7738, \; d_2 = 27.6229
     $$
     c) 算$x_0$的$1-\alpha$**置信区间**$(d_1 + \overline x, d_2 + \overline x)$
     $$
     置信区间：(d_1+\overline x,d_2+\overline x) = (170.3988,181.2479)
     $$
   
   - **控制**
   
     当$y \in (85,95)$时，求$x$的$0.95$的控制区间
   
     a) 查**U值表**得$u_{1-\alpha/2}$

     当**显著性水平**$\alpha=0.05$时，$u_{0.975}=1.96$

     b) 计算$x_1$和$x_2$（$y_1$为给定区间下界，$y_2$为给定区间上界）
     
     注意：仅当$n$足够大且$x$与$\overline x$接近时成立
     $$
     x_1 = \frac{1}{\widehat \beta_1} (y_1 + \widehat \sigma_e u_{1 - \alpha/2} - \widehat \beta_0)  = \frac{1}{0.719}(85+1.329 \times 1.96 - (-16.07)) = 144.186 \\
     x_2 = \frac{1}{\widehat \beta_1} (y_2 - \widehat \sigma_e u_{1-\alpha/2} - \widehat \beta_0) = \frac{1}{0.719}(95 - 1.329 \times 1.96 - (-16.07)) = 150.862
     $$
     c) 得到**控制区间**
   
     当$\widehat \beta_1 =0.719> 0$时，$x$的$0.95$的控制区间为$(x_1,x_2)=(144.186,150.862)$

###### 4.3.2 多元回归

| $x_1$ | -1   | -1   | -1   | -1   | 1    | 1    | 1    | 1    |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| $x_2$ | -1   | -1   | 1    | 1    | -1   | -1   | 1    | 1    |
| $x_3$ | -1   | 1    | -1   | 1    | -1   | 1    | -1   | 1    |
| $y$   | 7.6  | 10.3 | 9.2  | 10.2 | 8.4  | 11.1 | 9.8  | 12.6 |

1. 解**回归方程**（最小二乘法）

   a) 建立$X'$
   $$
   X' = 
   \begin{bmatrix}
   1 & 1 & 1 & 1 & 1& 1 & 1 & 1\\
   -1  & -1   & -1  & -1   & 1   & 1    & 1   & 1    \\
   -1  & -1   & 1   & 1    & -1  & -1   & 1   & 1    \\
   -1  & 1    & -1  & 1    & -1  & 1    & -1  & 1  
   \end{bmatrix}
   $$
   b) 计算$X'X$
   $$
   X'X = 
   \begin{bmatrix}
   8 & 0 & 0 & 0 \\
   0 & 8 & 0 & 0 \\
   0 & 0 & 8 & 0 \\
   0 & 0 & 0 & 8
   \end{bmatrix}
   $$
   c) 计算$C$
   $$
   C = L^{-1} = (X'X)^{-1} =
   \begin{bmatrix}
   0.1250 & 0                 & 0                 & 0                 \\
   0                 & 0.1250 & 0                 & 0                 \\
   0                 & 0                 & 0.1250 & 0                 \\
   0                 & 0                 & 0                 & 0.1250
   \end{bmatrix}
   $$
   d) 计算$X'Y$
   $$
   X'Y = 
   \begin{bmatrix}
   79.2 \\
   4.6  \\
   4.4  \\
   9.2
   \end{bmatrix}
   $$
   e) 计算$\widehat \beta$
   $$
   \widehat \beta = 
   \begin{bmatrix}
   \widehat \beta_0 \\
   \widehat \beta_1 \\
   \widehat \beta_2 \\
   \widehat \beta_3
   \end{bmatrix} 
    = C(X'Y) 
    = 
    \begin{bmatrix}
   9.9000 \\
   0.5750\\
   0.5500\\
   1.1500
    \end{bmatrix}
   $$
   f) 得到**回归方程**
   $$
   \widehat y = \widehat \beta_0 + \widehat \beta_1 x_1 + \widehat \beta_2 x_2 + \widehat \beta_3 x_3 = 9.9 + 0.5750x_1 + 0.55x_2 + 1.15x_3
   $$

2. **检验线性关系显著性**

   a) 检验**整体**线性关系（即检验假设$H_0: \beta_1 = \beta_2 = ... = \beta_k = 0$）
   $$
   S_总 = \sum_{t = 1}^8 (y_t - 9.9)^2 = \sum_{t = 1}^n y_t^2 -8 \times 9.9^2  = 801.1-8 \times 9.9^2 = 17.02\\
   U_R = \sum_{t = 1}^8 (\widehat \beta_1 x_{t1} + \widehat \beta_2 x_{t2} + \widehat \beta_3 x_{t3})^2 = 15.645\\
   Q_e = S_总 - U_R  = 17.02 - 15.645 = 1.375\\
   F = \frac{U_R /k}{Q_e/(n - k -1)} = \frac{15.645 /3}{1.375/(8 - 3 -1)} = 15.1709
   $$
   $F=15.1709 > 6.59 = F_{0.95}(k,n-k-1)$，否定$H_0$，即认为$\beta_1,\beta_2,\beta_3$不全为零，数据**具有线性**

   b) 检验**回归系数**
   $$
   C = L^{-1} = [c_{ij}] \\
   F_i = \frac{U_i}{Q_e / (n - k - 1)} = \frac{\widehat \beta_i^2 / c_{ii}}{Q_e / (n - k - 1)} \\
   F_1 = 7.6945, \; F_2 = 7.0400 , \; F_3 = 30.7782
   $$
   对**显著性水平**$\alpha=0.05$查**F分布表**得临界值$F_{0.95}(1,4)=7.71$

   情况二：存在$F_i < F_{0.95}(1,4),i = 1,2$，剔除$F_i$最小值对应的自变量$x_2$，建立新的回归方程

1. 建立新的**回归方程**
   $$
   \widehat y = \widehat \beta_0 + \widehat \beta_1 x_1  + \widehat \beta_3 x_3 = 9.9 + 0.575x_1 + 1.15x_3
   $$

2. **检验线性关系显著性**

   a) 检验**整体**线性关系

   由于第一次已检验过，故不必再次检验

   b) 检验**回归系数**
   $$
   C = L^{-1} = [c_{ij}] \\
   F_i = \frac{U_i}{Q_e / (n - k - 1)} = \frac{\widehat \beta_i^2 / c_{ii}}{Q_e / (n - k - 1)} \\
   F_1 = 3.4848,\; F_3 = 13.9394
   $$
   对**显著性水平**$\alpha=0.05$查**F分布表**得临界值$F_{0.95}(1,5)=6.61$

   情况二：若存在$F_1 < F_{0.95}(1,5)$，剔除对应的自变量$x_1$，建立新的回归方程

1. 建立新的**回归方程**
   $$
   \widehat y = \widehat \beta_0 + \widehat \beta_3 x_3 = 9.9  + 1.15x_3
   $$

2. **检验线性关系显著性**

   a) 检验**整体**线性关系

   由于第一次已检验过，故不必再次检验

   b) 检验**回归系数**
   $$
   C = L^{-1} = [c_{ij}] \\
   F_i = \frac{U_i}{Q_e / (n - k - 1)} = \frac{\widehat \beta_i^2 / c_{ii}}{Q_e / (n - k - 1)} \\
   F_3 = 9.8671
   $$
   对**显著性水平**$\alpha=0.05$查**F分布表**得临界值$F_{0.95}(1,6)=5.99$

   情况一：$F_3 > F_{0.95}(1,6)$，则原回归方程即最终回归方程
   $$
   \widehat y = \widehat \beta_0 + \widehat \beta_3 x_3 = 9.9  + 1.15x_3
   $$

##### 4.4 代码实现

###### 4.4.1 简单回归

 [LP.m](LP.m) 

代码：

```matlab
clc, clear
% 读取数据
X = readmatrix('/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/dataset.xlsx');
disp('自变量：');
disp(X);
Y = readmatrix('/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/dataset.xlsx','Sheet','Sheet2');
disp('因变量：');
disp(Y);
% 样本数量n
n = size(Y,1);
% 解平均值
x_avg = sum(X(:,2))/n;
y_avg = sum(Y)/n;
% 解方差
Lxy = 0;
Lxx = 0;
Lyy = 0;
for t=1:n
    Lxy = Lxy + (X(t,2) - x_avg) * (Y(t,1) - y_avg);
    Lxx = Lxx + (X(t,2) - x_avg)^2;
    Lyy = Lyy + (Y(t,1) - y_avg)^2;
end
% 回归系数
b1 = Lxy / Lxx;
b0 = y_avg - x_avg * b1;
disp('系数b0：');
disp(b0);
disp('系数b1：');
disp(b1);

% 分割线
disp(' ');
disp('----------------- 分割线 ----------------- ');
disp(' ');

% 显著性检验
UR = b1^2 * Lxx;
disp('回归平方和UR：');
disp(UR);
Qe = Lyy - UR;
disp('残差平方和Qe：');
disp(Qe);
F = UR * (n - 2) / Qe;
disp('F：');
disp(F);

% 分割线
disp(' ');
disp('----------------- 分割线 ----------------- ');
disp(' ');

% 预测
% 计算预测值
x0 = input('请输入待预测的自变量的值：');
y_exp = b0 + b1 * x0;
disp('预测值为：');
disp(y_exp);
% 计算预测区间
sigma = sqrt(Qe / (n - 2));
t = input('请输入t的取值：');
delta = sigma * t * sqrt(1 + 1/n + (x0-x_avg)^2/Lxx);
y1 = y_exp + delta;
y2 = y_exp - delta;
disp('预测区间下界为：');
disp(y2);
disp('预测区间上界为：');
disp(y1);

% 反预测
y0 = input('请输入待反预测的值：');
sigma2 = Qe / (n - 2);
F_theory = input('请输入F的临界值：');
a = b1^2-sigma2*F_theory/Lxx;
b = -2*b1*(y0-y_avg);
c = (y0-y_avg)^2-sigma2*F_theory*(1+1/n);
d1 = (sqrt(b^2 - 4*a*c) - b) / (2*a);
d2 = (- sqrt(b^2 - 4*a*c) - b) / (2*a);
x1 = x_avg + d1;
x2 = x_avg + d2;
disp('区间边界为：');
disp(x1);
disp(x2);

% 控制
u = input('请输入u的取值：');
sigma = sqrt(Qe / (n - 2));
y1_control = input('请输入控制区间下界：');
y2_control = input('请输入控制区间上界：');
x1_control = 1/b1 * (y1_control+sigma*u-b0);
x2_control = 1/b1 * (y2_control-sigma*u-b0);
disp('区间边界为：');
disp(x1_control);
disp(x2_control);
```

结果：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/简单回归结果1.png" alt="简单回归结果1" style="zoom:33%;" />

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/简单回归结果2.png" alt="简单回归结果2" style="zoom:33%;" />

###### 4.4.2 多元回归

 [LP2.m](LP2.m) 

代码：

```matlab
% 读取数据，建立回归方程
clc, clear
X = readmatrix('/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/dataset2.xlsx');
disp('自变量：');
disp(X);
L = X' * X;
C = inv(L);
Y = readmatrix('/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/dataset2.xlsx','Sheet','Sheet2');
disp('因变量：');
disp(Y);
A = X' * Y;
B = C * A;
disp('线性回归系数矩阵：');
disp(B);

% 样本数，自变量参数
n = size(Y,1);
k = size(X,2) - 1;

% 计算S总
y_avg = sum(Y)/n;
y2_sum = sum(Y.^2);
S = y2_sum - n * y_avg^2;
disp('S总：');
disp(S);

% 计算UR
UR = 0;
for t=1:n
    temp = 0;
    for i=1:k
        temp = temp + B(i+1,1) * X(t,i+1);
    end
    UR = UR + temp^2;
end
disp('回归平方和UR：');
disp(UR);

% 计算Qe
Qe = S - UR;
disp('残差平方和Qe：');
disp(Qe);

% 计算F
F = UR * (n - k - 1) / Qe / k;
disp('F总：');
disp(F);

% 检验回归系数
f = zeros(1,k);
for i=1:k
    f(i) = B(i+1,1)^2 / C(i,i) / Qe * (n - k - 1);
end
disp('回归系数F(i)：');
disp(f);
```

第一次回归：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/多元回归结果1.png" alt="多元回归结果1" style="zoom:33%;" />

第二次回归：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/多元回归结果2.png" alt="多元回归结果2" style="zoom:33%;" />

第三次回归：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/多元回归结果3.png" alt="多元回归结果3" style="zoom:33%;" />

#### 5. 参考资料

1. 《应用数理统计》（孙荣恒）P230-P263 [应用数理统计（孙荣恒）.pdf](../../教材/应用数理统计（孙荣恒）.pdf) 
2. [Mooc：Python机器学习（北京理工大学）【第二周】单元5：回归-线性回归](https://www.icourse163.org/learn/BIT-1001872001?tid=1001965001#/learn/content?type=detail&id=1002863720&cid=1003262094)
2. $t$分布分位数表

![t分布1](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/t分布1.png)

![t分布2](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/t分布2.png)

4. 正态分布常用分位数表

![正态分布常用分位数表](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/正态分布常用分位数表.png)

5. F分布表

![F分布1](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/F分布1.png)

![F分布2](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/F分布2.png)

![F分布3](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/F分布3.png)

![F分布4](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/F分布4.png)

![F分布5](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/F分布5.png)

![F分布6](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/F分布6.png)

![F分布7](/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-回归-线性回归【hxy】/F分布7.png)