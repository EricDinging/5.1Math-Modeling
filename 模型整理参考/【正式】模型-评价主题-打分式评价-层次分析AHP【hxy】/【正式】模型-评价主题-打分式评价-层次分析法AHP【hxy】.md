[TOC]

### 模型-评价主题-打分式评价-层次分析法AHP【hxy】

#### 1. 模型名称

层次分析法（Analytic Hierarchy Process, AHP）

#### 2. 适用范围

适用于具有**分层交错评价指标的目标系统**，而且**目标值难以定量描述**

#### 3. 形式

具有**定性**的**目标层**，**准则层**和**方案层**

#### 4. 求解过程

##### 4.1 概念

将与决策有关的元素分解成目标、准则、方案等层次，在此基础之上进行定性和定量分析的决策方法进行定性和定量分析的决策方法

##### 4.2 步骤

1. 将评价体系**分层**，分为**目标**、**准则**、**方案**

   <img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-层次分析AHP【hxy】/分层概览.png" alt="分层概览" style="zoom:50%;" />

2. 构造各层次中所有**判别矩阵**[^1]

   **判别矩阵**$A-C$

   | $A$   | $C_1$ | $C_2$ | ···  | $C_n$ |
   | ----- | ----- | ----- | ---- | ----- |
   | $C_1$ |       |       |      |       |
   | $C_2$ |       |       |      |       |
   | ···   |       |       |      |       |
   | $C_n$ |       |       |      |       |

   **判别矩阵**$C_i-P(i=1,2,...,n)$

   | $C_i$ | $P_1$ | $P_2$ | ···  | $P_m$ |
   | ----- | ----- | ----- | ---- | ----- |
   | $P_1$ |       |       |      |       |
   | $P_2$ |       |       |      |       |
   | ···   |       |       |      |       |
   | $P_m$ |       |       |      |       |

   具体数值根据**矩阵评分比例标度表**由专家确定：

   | 因素$i$比因素$j$   | 量化值     |
   | ------------------ | ---------- |
   | 同等重要           | 1          |
   | 稍微重要           | 3          |
   | 较强重要           | 5          |
   | 强烈重要           | 7          |
   | 极端重要           | 9          |
   | 两相邻判断的中间值 | 2，4，6，8 |

3. **层次单排序**：得到**权向量$W^{(1)}$**并进行**一致性检验**

   a) 根据**判别矩阵$A-C$**，计算**最大特征值$\lambda_{max}$**

   b) 计算**一致性指标**$CI$
   $$
   CI = \frac{\lambda_{max} - n}{n - 1}
   $$
   c) 查表得到$n$情况下**随机一致性指标**$RI(n)$

   | $n$  | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   |
   | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
   | $RI$ | 0    | 0    | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 | 1.51 |

   d) 计算**层次单排序的一致性比率**$CR$
   $$
   CR = \frac{CI}{RI(n)}
   $$
   当$CR < 0.1$时，满足一致性条件，模型有效

   e) 对其对应的**特征向量**$u$进行**归一化**得到**权向量$W^{(1)}$**
   $$
   w_i^{(1)} = \frac{u_i}{\sum_{j=1}^n u_j} \quad i = 1,2,...,n
   $$
   
4. **层次组合排序**：得到**组合权重$W^{(3)}$**并进行**一致性检验**

   a) 根据**判别矩阵$C_i - P$**，计算**最大特征根$\lambda_i(i=1,2,...,n)$**

   b) 对其对应的**特征向量**$u_i$进行**归一化**得到**权向量$W^{(2)}$**
   $$
   W^{(2)} = (W_1^{(2)}\; W_2^{(2)} \; ... \;  W_n^{(2)})
   $$
   c) 计算**一致性指标**$CI_i(i=1,2,...,n)$
   $$
   CI_i = \frac{\lambda_i - m}{m - 1}
   $$
   d) 查表得到$m$情况下**随机一致性指标**$RI(m)$
   
   e) 计算**层次组合排序的一致性比率**$CR$（当$RI_1 = RI_2 = ··· = RI_n = RI(m)$时)
   $$
   CR = \frac{w_1CI_1+w_2CI_2+···+w_nCI_n}{w_1RI_1+w_2RI_2+···+w_nRI_n} = \frac{w_1CI_1+w_2CI_2+···+w_nCI_n}{RI(m)}
   $$
   
   当$CR < 0.1$时，满足一致性条件，模型有效
   
   f) 计算$P_i(i=1,2,...,m)$对$A$的**组合权重$W^{(3)}$**
   $$
   W^{(3)} = W^{(2)} · W^{(1)}
   $$
   
5. 根据**组合权重$W^{(3)}$**，数值最大的为最佳方案

##### 4.3 实例

$P_1$是苏杭，$P_2$是北戴河，$P_3$是桂林，根据景色、费用、居住、饮食和旅途选择旅游地

1. 将评价体系**分层**，分为**目标**、**准则**、**方案**

   <img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-评价主题-打分式评价-层次分析AHP【hxy】/实例分层.png" alt="实例分层" style="zoom: 67%;" />

2. 构造各层次中所有**判别矩阵**[^1]
   $$
   AC = 
    \begin{bmatrix}
      1 & 1/2 & 4 & 3 & 3 \\
      2 & 1 & 7 & 5 & 5 \\
      1/4 & 1/7 & 1 & 1/2 & 1/3 \\
      1/3 & 1/5 & 2 & 1  & 1\\
      1/3 & 1/5 & 3 & 1 & 1 \\
     \end{bmatrix} \quad
     CP_1 = 
    \begin{bmatrix}
      1 & 2 & 5  \\
      1/2 & 1 & 2 \\
      1/5 & 1/2 & 1  \\
     \end{bmatrix}\quad
     CP_2 = 
    \begin{bmatrix}
      1 & 1/3 & 1/8  \\
      3 & 1 & 1/3 \\
      8 & 3 & 1  \\
     \end{bmatrix} \\
     CP_3 = 
    \begin{bmatrix}
      1 & 1 & 3  \\
      1 & 1 & 3 \\
      1/3 & 1/3 & 1  \\
     \end{bmatrix} \quad
     CP_4 = 
    \begin{bmatrix}
      1 & 3 & 4  \\
      1/3 & 1 & 1 \\
      1/4 & 1 & 1  \\
     \end{bmatrix}\quad
      CP_5 = 
    \begin{bmatrix}
      1 & 1 & 1/4  \\
      1 & 1 & 1/4 \\
      4 & 4 & 1  \\
     \end{bmatrix}
   $$

3. **层次单排序**：得到**权向量$w$**并进行**一致性检验**

   a) 根据**判别矩阵$AC$**，计算**最大特征值$\lambda_{max}$**
   $$
   \lambda_{max} = 5.073
   $$
   b) 计算**一致性指标**$CI$
   $$
   CI = \frac{\lambda_{max} - n}{n - 1} = \frac{5.073 - 5}{5 - 1} = 0.018
   $$
   c) 查表得$n$情况下**随机一致性指标**$RI(n)$
   $$
   RI(n) = 1.12
   $$
   d) 计算**层次单排序的一致性比率**$CR$
   $$
   CR = \frac{0.018}{1.12} = 0.016 < 0.1
   $$
   满足一致性条件，模型有效

   e) 得到**权向量$W^{(1)}$**
   $$
   W^{(1)} = \begin{bmatrix}
   0.263 \\0.475\\0.055\\0.090\\0.110
   \end{bmatrix}
   $$
   
4. **层次组合排序**：得到**组合权重$W^{(3)}$**并进行**一致性检验**

   a) 根据**判别矩阵$CP_i$**，计算**最大特征根$\lambda_i(i=1,2,...,n)$**
   $$
   \lambda_1 = 3.005, \; \lambda_2 = 3.003, \; \lambda_3 = 3, \; \lambda_4 = 3.009, \; \lambda_5 = 3, \;
   $$
   b) 得到**权向量$W^{(2)}$**
   $$
   W_1^{(2)} = 
    \begin{bmatrix}
      0.595 \\
      0.277  \\
      0.129 \\
     \end{bmatrix}
     W_2^{(2)} = 
    \begin{bmatrix}
      0.082 \\
      0.236  \\
      0.682 \\
     \end{bmatrix}
     W_3^{(2)} = 
    \begin{bmatrix}
      0.429 \\
      0.429  \\
      0.149 \\
     \end{bmatrix}
     W_4^{(2)} = 
    \begin{bmatrix}
      0.633 \\
      0.193  \\
      0.175 \\
     \end{bmatrix}
     W_5^{(2)} = 
    \begin{bmatrix}
      0.166 \\
      0.166  \\
      0.668 \\
     \end{bmatrix} \\
   W^{(2)} = 
    \begin{bmatrix}
      0.595 & 0.082 & 0.429 & 0.633 & 0.166  \\
      0.277 & 0.236 & 0.429 & 0.193 & 0.166  \\
      0.129 & 0.682 & 0.142 & 0.175 & 0.668  \\
     \end{bmatrix}
   $$
   c) 计算**一致性指标**$CI_i(i=1,2,...,n)$
   $$
   CI_1 = \frac{\lambda_1 - m}{m - 1} = \frac{3.005 - 3}{3 - 1} = 0.003, \; CI_2 = 0.001, \; CI_3 = 0, \; CI_4 = 0.005, \; CI_5 = 0
   $$
   d) 查表得到$m$情况下**随机一致性指标**$RI(m)$
   $$
   RI(m) = RI(3) = 0.58
   $$
   e) 计算**层次组合排序的一致性比率**$CR$
   $$
   CR = \frac{w_1CI_1+w_2CI_2+···+w_nCI_n}{RI(m)} \\ = \frac{0.263 \times 0.003 + 0.475 \times 0.001 + 0.055 \times 0 + 0.090 \times 0.005 + 0.110 \times 0}{0.58} = 0.015 < 0.1
   $$
   满足一致性条件，模型有效

   f) 计算$P_i(i=1,2,...,m)$对$A$的**组合权重$W^{(3)}$**
   $$
   W^{(3)} = W^{(2)} · W^{(1)} = 
   \begin{bmatrix}
      0.595 & 0.082 & 0.429 & 0.633 & 0.166  \\
      0.277 & 0.236 & 0.429 & 0.193 & 0.166  \\
      0.129 & 0.682 & 0.142 & 0.175 & 0.668  \\
     \end{bmatrix} · 
     \begin{bmatrix}
   0.263 \\0.475\\0.055\\0.090\\0.110
   \end{bmatrix} \\
   = \begin{bmatrix}
   0.595 \times 0.263 + 0.082 \times 0.475 + 0.429 \times 0.055 + 0.633 \times 0.09 + 0,166 \times 0.110 \\
   0.277 \times 0.263 + 0.236 \times 0.475 + 0.429 \times 0.055 + 0.193 \times 0.090 + 0.166 \times 0.110\\
   0.129 \times 0.263 + 0.682 \times 0.475 + 0.142 \times 0.055 + 0.175 \times 0.090 + 0.668 \times 0.110
   \end{bmatrix} 
   =  \begin{bmatrix}
   0.300 \\0.246\\0.456
   \end{bmatrix} \\
   $$

5. 根据**组合权重$W^{(3)}$**，数值最大的为最佳方案
   $$
   W^{(3)} = 
   \begin{bmatrix}
   0.300 \\0.246\\0.456
   \end{bmatrix} \\
   $$
   $P_3$对$A$的**组合权重$W_3^{(3)} = 0.456$**最大，$P_3$桂林为最佳决策

##### 4.4 代码实现

 [ahp.m](ahp.m) 

Matlab自定义函数：``app.m``

```matlab
% 给定待求矩阵Matrix，矩阵下一层的样本数量number，该样本数量下的RI
% 返回权向量weight和CI
function [weight, CI]=ahp(Matrix,number,RI)

% 打印待求矩阵Matrix
disp('待求矩阵为：');
disp(Matrix);

% 求解权向量
% 计算AC的特征值V和特征向量U
[U,V] = eig(Matrix);
% 找到最大特征值V_max
Max_eig = max(max(V));
disp('其最大特征值为：'); disp(Max_eig);
% 找到V_max对应的特征向量u
[r,c] = find(V == Max_eig,1);
% 将u归一化得到权向量W(1)
disp('其特征值法求权重的结果为：');
weight = U(:,c) ./ sum(U(:,c));
disp(weight);

% 进行一致性检验
% 得到CI
CI = (Max_eig - number)/(number - 1);
disp('其一致性指标CI='); disp(CI);
% 得到CR
CR = CI/RI;
disp('其一致性比例CR='); disp(CR);
% 得出一致性检验结果
if CR < 0.10
    disp('因为CR<0.10，所以该判断矩阵的一致性可以接受！');
else
    disp('注意：CR>=0.10，因此该判断矩阵A需要进行修改！');
end

% 分割线
disp(' ');
disp('------------------------ 分割线 ------------------------');
disp(' ');
```

 [main_ahp.m](main_ahp.m) 

Matlab主程序：

```matlab
clc, clear
% 输入
% 输入AC, CP1, CP2, CP3, CP4, CP5 
AC = [1 1/2 4 3 3;
    2 1 7 5 5;
    1/4 1/7 1 1/2 1/3;
    1/3 1/5 2 1 1;
    1/3 1/5 3 1 1];
CP1 = [1 2 5;
     1/2 1 2;
     1/5 1/2 1];
CP2 = [1 1/3 1/8;
      3 1 1/3;
      8 3 1];
CP3 = [1 1 3;
     1 1 3;
     1/3 1/3 1];
CP4 = [1 3 4;
     1/3 1 1;
     1/4 1 1];
CP5 = [1 1 1/4;
     1 1 1/4;
     4 4 1];
% 输入n, m    
n = 5; 
m = 3;
% 输入RI5, RI3
RI5 = 1.12;
RI3 = 0.58;

% 计算
% 计算AC, CP1, CP2, CP3, CP4, CP5的权向量和CI（调用ahp.m自定义函数）
CP_CI = [];
[AC_weight, AC_CI] = ahp(AC,n,RI5);
[CP1_weight, CP_CI(1)] = ahp(CP1,m,RI3);
[CP2_weight, CP_CI(2)] = ahp(CP2,m,RI3);
[CP3_weight, CP_CI(3)] = ahp(CP3,m,RI3);
[CP4_weight, CP_CI(4)] = ahp(CP4,m,RI3);
[CP5_weight, CP_CI(5)] = ahp(CP5,m,RI3);
% 计算组合CR_combine
CR_combine = 0;
for i = 1:5
  CR_combine = CR_combine + AC_weight(i,1)*CP_CI(i);
end
disp('组合CR为：')
disp(CR_combine);
if CR_combine < 0.10
    disp('因为CR<0.10，所以该判断矩阵的一致性可以接受！');
    disp(' ');
else
    disp('注意：CR>=0.10，因此该判断矩阵A需要进行修改！');
    disp(' ');
end
% 计算组合权向量weight_combine
weight_combine = [CP1_weight CP2_weight CP3_weight CP4_weight CP5_weight] * AC_weight;
disp('组合权向量W(3)为：');
disp(weight_combine);
```

结果：

AC

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-层次分析AHP【hxy】/AC.png" alt="AC" style="zoom:50%;" />

CP1

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-层次分析AHP【hxy】/CP1.png" alt="CP1" style="zoom:50%;" />

CP2

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-层次分析AHP【hxy】/CP2.png" alt="CP2" style="zoom:50%;" />

CP3

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-层次分析AHP【hxy】/CP3.png" alt="CP3" style="zoom:50%;" />

CP4

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-层次分析AHP【hxy】/CP4.png" alt="CP4" style="zoom:50%;" />

CP5

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-层次分析AHP【hxy】/CP5.png" alt="CP5" style="zoom:50%;" />

组合

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-层次分析AHP【hxy】/组合.png" alt="组合" style="zoom:50%;" />

#### 5. 参考资料

1. [层次分析法详细步骤](https://blog.csdn.net/mmm_jsw/article/details/84863416)
2. [层次分析法详细讲解（小白必看&电脑查看）](https://blog.csdn.net/fencecat/article/details/112284913?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-4.pc_relevant_paycolumn_v2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-4.pc_relevant_paycolumn_v2&utm_relevant_index=7)
3. 高晓沨第三次培训PPT：P24 - P27
4. 上文提到的部分名词解释

[^1]: 利用Saaty在1970年提出的Saaty Table，对各因素进行成对的重要性判定，建立判别矩阵

