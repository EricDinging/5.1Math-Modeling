[TOC] 

### 模型-评价主题-打分式评价-数据包络分析法【hxy】

#### 1. 模型名称

数据包络分析法（Data Envelopment Analysis，DEA）

#### 2. 适用范围

根据多项投入指标和多项产出指标，对具有可比性的同类型单元（称为决策单元DMU）进行有效性评价，最初用于一些非赢利部门（如教育、卫生、政府机构）的运转的有效性的评价，后来被用于更广泛的领域（如金融、经济、项目评估等）

例如：大学一个系的投入包括教师、教师的工资、办公经费、文献资料费等，产出包括培养本科生和研究生、发表的论文、完成的科研项目等。DEA可以对若干个同类型的这种部门或单位（它们有相同的目标和任务，有相同的输入和输出指标，有相同的外部环境）进行相对有效性的评价。

#### 3. 形式

多项投入指标，多项产出指标，有同类型决策单元DMU

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-数据包络分析法【hxy】/形式.png" alt="形式" style="zoom:50%;" />

#### 4. 求解过程

##### 4.1 步骤

1. 解得$DMU_i$的**最佳权向量**$\omega_i^*, \; \mu_i^*$及**最佳权向量**时**效率评价指数$E_{ii}$**

   对每一个$DMU_i$，解以下**极大化问题**（对于$E_{ii}$公式推导过程，见参考资料）

$$
\max E_{ii} = \max \frac{y_i^T u}{x_i^T v} \\
s.t. \begin{cases}
\frac{y_j^T u}{x_j^T v} \leq 1, \; j = 1,2,...,n \\
u \geq 0, \; v \geq 0\\
\end{cases} \\
where \quad x_i = (x_{1i},x_{2i},...,x_{mi})^T, \quad y_i = (y_{1i},y_{2i},...,y_{si})^T \\
input \; weight \; vector: \; v = (v_1,v_2,...,v_n)^T \\
output \; weight \; vector: \; u = (u_1,u_2,...,u_n)^T \\
$$

​		此问题是**分式规划问题**，令
$$
t = \frac{1}{x_i^T v}, \; \omega = tv, \; \mu = tu
$$
​		转化为解等价的**线性规划问题$P$**（**Charnes-Cooper变换**）
$$
\max E_{ii} = \max y_i^T \mu \\
s.t. \begin{cases}
-x_j^T \omega + y_j^T \mu \leq 0, \; j = 1,2,...,n \\
x_i^T \omega = 1 \\
\omega \geq 0, \; \mu \geq 0\\
\end{cases}
$$
​		解得$DMU_i$的**最佳权向量**$\omega_i^*, \; \mu_i^*$及**最佳权向量**时**效率评价指数$E_{ii}$**

2. 检验DEA的**有效性**

**方法一**

- **CASE1** $E_{ii} \neq 1$：**非DEA有效**
- **CASE2** $E_{ii} = 1$：**弱DEA有效($C^2R$)**
- **CASE3** 存在$\omega_i^* > 0, \; \mu_i^* > 0$ 且 $E_{ii} = 1$：**DEA有效($C^2R$)**

**方法二(常用)**

解**$P$的对偶模型的等式形式$D_\epsilon$**
$$
\min \left(\theta-\varepsilon\left(e_{1}^{T} s^{-}+e_{2}^{T} s^{+}\right)\right)\\
s.t. \left\{\begin{array}{l}
\sum_{j=1}^{n} \lambda_{j} x_{j}+s^{-}=\theta x_{i},  \; \sum_{j=1}^{n} \lambda_{j} y_{j}-s^{+}=y_{i}, \\
\lambda \geq 0, s^{-} \geq 0, s^{+} \geq 0
\end{array}\right. \\ \\
\epsilon = 10^{-6}(一个很小的正数), \\
m项输入的松弛变量：s^{-}=\left(s_{1}^{-}, s_{2}^{-}, \cdots, s_{m}^{-}\right), \\
s项输出的松弛变量：s^+ = \left(s_{1}^{+}, s_{2}^{+}, \cdots, s_{s}^{+}\right), \\
n个DMU的 组合系数：\lambda=\left(\lambda_{1}, \lambda_{2}, \cdots, \lambda_{n}\right), \\
e_{1}^{T}=(1,1, \cdots, 1)_{1 \times m}, \; e_{2}^{T}=(1,1, \cdots, 1)_{1 \times s} \\
$$

- **CASE1** $\theta^* \neq 1$：**非DEA有效**
- **CASE2** $\theta^* = 1$：**弱DEA有效($C^2R$)**
- **CASE3** 存在$s^{*-} > 0, \; s^{*+} > 0$ 且 $\theta^* = 1$：**DEA有效($C^2R$)**

##### 4.2 实例

|                           | $DMU_1$ | $DMU_2$ | $DMU_3$ | $DMU_4$ | $DMU_5$ |
| ------------------------- | ------- | ------- | ------- | ------- | ------- |
| 投入-教职工（人）         | 60      | 70      | 85      | 106     | 35      |
| 投入-教职工工资（万元）   | 156     | 200     | 157     | 263     | 105     |
| 投入-运转经费（万元）     | 50      | 180     | 100     | 86      | 30      |
| 产出-毕业的本科生（人）   | 80      | 60      | 90      | 96      | 30      |
| 产出-毕业的研究生（人）   | 12      | 13      | 20      | 17      | 8       |
| 产出-发表的论文（篇）     | 27      | 25      | 15      | 28      | 3       |
| 产出-完成的科研项目（项） | 4       | 2       | 5       | 5       | 1       |

1. 解得$DMU_i$的**最佳权向量**$\omega_i^*, \; \mu_i^*$及**最佳权向量**时**效率评价指数$E_{ii}$**

   |          | $DMU_1$ | $DMU_2$ | $DMU_3$ | $DMU_4$ | $DMU_5$ |
   | -------- | ------- | ------- | ------- | ------- | ------- |
   | $\omega$ | 0.0167  | 0.0143  | 0.0118  | 0.0000  | 0.0263  |
   | $\omega$ | 0.0000  | 0.0000  | 0.0000  | 0.0014  | 0.0000  |
   | $\omega$ | 0.0000  | 0.0000  | 0.0000  | 0.0073  | 0.0026  |
   | $\mu$    | 0.0125  | 0.0000  | 0.0000  | 0.0000  | 0.0000  |
   | $\mu$    | 0.0000  | 0.0554  | 0.0235  | 0.0442  | 0.1250  |
   | $\mu$    | 0.0000  | 0.0071  | 0.0000  | 0.0000  | 0.0000  |
   | $\mu$    | 0.0000  | 0.0000  | 0.1059  | 0.0138  | 0.0000  |
   | $E_{ii}$ | 1.0000  | 0.8982  | 1.0000  | 0.8206  | 1.0000  |
   
2. 检验DEA的**有效性**

   采用**方法二**

   |             | $DMU_1$ | $DMU_2$  | $DMU_3$ | $DMU_4$ | $DMU_5$ |
   | ----------- | ------- | -------- | ------- | ------- | ------- |
   | $\lambda^*$ | 1.0000  | 0.8472   | 0.0000  | 1.0964  | 0.0000  |
   | $\lambda^*$ | 0.0000  | 0.0000   | 0.0000  | 0.0000  | 0.0000  |
   | $\lambda^*$ | 0.0000  | 0.1417   | 1.0000  | 0.0536  | 0.0000  |
   | $\lambda^*$ | 0.0000  | 0.0000   | 0.0000  | 0.0000  | 0.0000  |
   | $\lambda^*$ | 0.0000  | 0.0000   | 0.0000  | 0.3464  | 1.0000  |
   | $s^{*-}$    | 0.0000  | 0.0000   | 0.0000  | 4.5215  | 0.0000  |
   | $s^{*-}$    | 0.0000  | 25.2345  | 0.0000  | 0.0000  | 0.0000  |
   | $s^{*-}$    | 0.0000  | 105.1508 | 0.0000  | 0.0000  | 0.0000  |
   | $s^{*+}$    | 0.0000  | 20.5278  | 0.0000  | 6.9272  | 0.0000  |
   | $s^{*+}$    | 0.0000  | 0.0000   | 0.0000  | 0.0000  | 0.0000  |
   | $s^{*+}$    | 0.0000  | 0.0000   | 0.0000  | 3.4454  | 0.0000  |
   | $s^{*+}$    | 0.0000  | 2.0972   | 0.0000  | 0.0000  | 0.0000  |
   | $\theta^*$  | 1.0000  | 0.8982   | 1.0000  | 0.8206  | 1.0000  |

   可知：

   - $DMU_1$：$\theta^* = 1$且$s^{*-} = 0, \; s^{*+} = 0$，满足**CASE3**，**DEA有效($C^2R$)**

   - $DMU_2$：$\theta^* \neq 1$，满足**CASE1**，**非DEA有效**

     根据**有效性的经济意义**，在不减少各项输出的前提下，构造一个新的$DMU_2$：
     $$
     DMU_2 = 0.8472 \times DMU_1 + 0.1417 \times DMU_3 \\
     = (62.8750,154.4083,56.5278,80.5278,13.0000,25.0000,4.0972)^T
     $$
     可以使$DMU_2$的投入按比例减少到原投入的$\theta_2^* = 0.8982$倍；由非零的松弛变量可知，可以进一步减少教职工工资$25.2345$万元、减少运转费用$105.1508$万元、多培养本科生$20$人，多完成$2$项科研项目

   - $DMU_3$：$\theta^* = 1$且$s^{*-} = 0, \; s^{*+} = 0$，满足**CASE3**，**DEA有效($C^2R$)**

   - $DMU_4$：$\theta^* \neq 1$，满足**CASE1**，**非DEA有效**

     根据**有效性的经济意义**，在不减少各项输出的前提下，构造一个新的$DMU_4$：
     $$
     DMU_4 = 1.0964 \times DMU_1 + 0.0536 \times DMU_3 + 0.3464 \times DMU_5
     $$
     可以使$DMU_4$的投入按比例减少到原投入的$\theta_4^* = 0.8206$倍；由非零的松弛变量可知，可以进一步减少教职工人数$4$人、多培养本科生$6$人，多发表$3$篇论文

   - $DMU_5$：$\theta^* = 1$且$s^{*-} = 0, \; s^{*+} = 0$，满足**CASE3**，**DEA有效($C^2R$)**

##### 4.3 代码实现

###### 1. Matlab解线性规划$P$

 [DEA1.m](DEA1.m) 

代码：

```matlab
clear
X=[ 60 70 85 106 35;
156 200 157 263 105;
50 180 100 86 30];        %用户输入多指标输入矩阵X
Y=[ 80 60 90 96 30;
12 13 20 17 8;
27 25 15 28 3;
4 2 5 5 1];        %用户输入多指标输出矩阵Y
%n为DMU数量，m为输入指标数量，n为输出指标数量
n=size(X',1);m=size(X,1);s=size(Y,1);
A=[-X'   Y'];
b=zeros(n,1);
LB=zeros(m+s,1);UB=[ ];
for i=1:n;
f=[zeros(1,m) -Y(:,i)'];
Aeq=[X(:,i)' zeros(1,s)];beq=1;
w(:,i)=linprog(f,A,b,Aeq,beq,LB,UB);    
%解线性规划，得DMUi的最佳权向量wi
E(i, i)=Y(:,i)'*w(m+1:m+s,i);            
%求出DMUi的相对效率值Eii
end
w                 %输出最佳权向量
E                 %输出相对效率值Eii
omega=w(1:m,:)     %输出投入权向量omega
mu=w(m+1:m+s,:)   %输出产出权向量mu 
```

结果：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-数据包络分析法【hxy】/结果1.png" alt="结果1" style="zoom:50%;" />

###### 2. Matlab解线性规划$D_\epsilon$

 [DEA2.m](DEA2.m) 

代码：

```matlab
clear
X=[ 60 70 85 106 35;
156 200 157 263 105;
50 180 100 86 30];        %用户输入多指标输入矩阵X
Y=[ 80 60 90 96 30;
12 13 20 17 8;
27 25 15 28 3;
4 2 5 5 1];        %用户输入多指标输出矩阵Y
n=size(X',1);m=size(X,1);s=size(Y,1);
epsilon=10^-10;     
%定义非阿基米德无穷小 =10^(－10)
f=[zeros(1,n) -epsilon*ones(1,m+s) 1];
A=zeros(1,n+m+s+1); b=0;
LB=zeros(n+m+s+1,1);UB=[ ];
LB(n+m+s+1)=-Inf;
for i=1:n;
Aeq=[X  eye(m)     zeros(m,s)   -X(:,i)
Y  zeros(s,m)   -eye(s)      zeros(s,1)];
beq=[zeros(m,1)
Y(:,i)];
w(:,i)= linprog (f,A,b,Aeq,beq,LB,UB);   
%解线性规划，得DMUi的最佳权向量wi
end
w                             %输出最佳权向量
lambda=w(1:n,:)          %输出 lambda*
s_minus=w(n+1:n+m,:)     %输出s*－
s_plus=w(n+m+1:n+m+s,:)  %输出s*＋
theta=w(n+m+s+1,:)        %输出 theta*
```

结果：

<img src="/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-评价主题-打分式评价-数据包络分析法【hxy】/结果2.png" alt="结果2" style="zoom:50%;" />

#### 5. 参考资料

1. [数模官网](https://anl.sjtu.edu.cn/mcm/docs/1%E6%A8%A1%E5%9E%8B/8%E8%AF%84%E4%BB%B7%E4%B8%BB%E9%A2%98/1%E6%89%93%E5%88%86%E5%BC%8F%E8%AF%84%E4%BB%B7/%E6%95%B0%E6%8D%AE%E5%8C%85%E7%BB%9C%E5%88%86%E6%9E%90%E6%B3%95/doc)

2. [Matlab解线性规划](https://blog.csdn.net/kswkly/article/details/90485816)

3. $E_{ii}的推导$

   a) DMU的输入和输出基础变量
   $$
   x_i = (x_{1i},x_{2i},...,x_{mi})^T, \quad y_i = (y_{1i},y_{2i},...,y_{si})^T \\
   Multi-index \; input \; matrix: \; X = (x_1,x_2,...,x_n)^T \\
   Multi-index \; output \; matrix: \; Y = (y_1,y_2,...,y_n)^T \\
   Input \; weight \; vector: \; v = (v_1,v_2,...,v_n)^T \\
   Output \; weight \; vector: \; u = (u_1,u_2,...,u_n)^T \\
   $$
   b) $DMU_i$的**总输入$I_i$**和**总输出$O_i$**
   $$
   I_i = (v_1x_{1i} + v_2x_{2i} + ... + v_mx_{mi}) = x_i^Tv \\
   O_i = (u_1y_{1i} + u_2y_{2i} + ... + u_sy_{si}) = y_i^Tu
   $$

   c) $DMU_i$的**效率评价指数$E_{ii}$**
   $$
   E_{ii} = \frac{O_i}{I_i} = \frac{y_i^T u}{x_i^T v}
   $$
