[TOC]

### 模型-预测主题-离散型预测-灰色预测 【czy】

#### 1. 模型名称

灰色预测模型 （Grey Prediction/Grey Forecast）

#### 2. 适用范围

数据少，看不出明显规律，适合灰色预测，利用微分方程挖掘数据本质规律

只适合短期预测，指数增长的预测

#### 3. GM(1,1)

##### 3.1 模型名称

一阶单变量灰色预测模型 （One Order Single Variable Grey Model, GM(1,1)）

  (1,1) : 一阶微分方程，一个变量

##### 3.2 求解步骤

![7ed0880c591458bf86a007d305340d8](1111111.assets/7ed0880c591458bf86a007d305340d8.jpg)

1. 建模最开始，需要进行数据的**级比检验**



（为了确定数据使用GM(1,1)模型的可行性，避免白忙活）

 计算
$$
\lambda(k)=\frac{x^{(0)}(k-1)}{x^{(0)}(k)},k=2,3,...,n
$$
如果$\lambda(k)$在区间$(e^{-\frac{2}{n+1}},e^{\frac{2}{n+2}})$内，说明可以用GM(1,1)模型

如果在区间外，可尝试平移变换，给每个数据都加上任意常数c，使其在区间内，求解后的数据再减去c

2. 制造**累加生成序列**



$x^{(0)}$是原始数据，$x^{(1)}$是对应的累加生成数据(如第3个数就是前3个原始数据的和)
$$
x^{(1)}(k)=\sum_{i=1}^kx^{(0)}(i)
$$
若生成的新序列$x^{(1)}$看起来像一个指数曲线(直线)，可构建一个微分方程来求解拟合曲线的函数表达式
$$
\frac{dx^{(1)}}{dt}+ax^{(1)}=u
$$
要预测下一年数值，就要求解微分方程，就要知道a和u(用最小二乘法)

但已知的数据是离散的非连续的，所以$\frac{dx^{(1)}}{dt}$写成$\frac{\Delta x^{(1)}}{\Delta t}$

而$\Delta t=(t+1)-t=1$,始终为1；而$\Delta x^{(1)}=x^{(1)}(t)-x^{(1)}(t-1)=x^{(0)}(t)$

得到方程$x^{(0)}(t)+ax^{(1)}(t)=u$

即
$$
x^{(0)}(t)=-ax^{(1)}(t)+u
$$
上式只有a和u两个未知数

3. $x^{(1)}(t)$修正为**均值生成序列**$z^{(1)}(t)$



考虑到原方程中有$\frac{\Delta x^{(1)}}{\Delta t}$,因此将$x^{(1)}(t)$改为取前后两个时刻的均值更合理
$$
z^{(1)}(t)=0.5x^{(1)}(t)+0.5x^{(1)}(t-1),t=2,...,n
$$
即方程改为
$$
x^{(0)}(t)=-az^{(1)}(t)+u
$$

4. 最小二乘法矩阵求解$Y=BU$

$$
 \left[
\matrix{
  x^{(0)}(2)\\
  x^{(0)}(3) \\
   \cdots\\
  x^{(0)}(N) 
}
\right]
 = \left[
\matrix{
 -\frac{1}{2}[x^{(1)}(2)+x^{(1)}(1)] & 1\\
 -\frac{1}{2}[x^{(1)}(3)+x^{(1)}(2)]& 1\\
  \cdots & 1\\
 -\frac{1}{2}[x^{(1)}(N)+x^{(1)}(N-1)] & 1 
}
\right]
\left[
\matrix{
  a \\

  u
}
\right]
$$

最小二乘法就是求$(Y-BU)^T(Y-BU)$取最小值时的U，

也就是拟合的函数的值与已知数据的平方差最小

$U$的估计值为(B,Y已知)
$$
\hat U=[\hat a,\hat u]^T=(B^TB)^{-1}B^TY
$$

5. 求解微分方程
   $$
   \hat x^{(1)}(k+1)=(x^{(0)}(1)-\frac{\hat b}{\hat a})e^{-\hat a k}+\frac{\hat b}{\hat a},k=0,1,...
   $$
   

6. **模型检验**----**残差检验**

看$\hat x^{(1)}(k+1)$求得的拟合值和实际值相差大不大

残差检验：
$$
ε(k)=\frac{x^{(0)}(k)-\hat x^{(0)}(k)}{x^{(0)}(k)},k=1,2,...,n
$$
其中$\hat x^{(0)}(1)=x^{(0)}(1)$

如果残差$ε(k)<0.2$,好。如果$ε(k)<0.1$,很好

##### 3.3 例子

![67835edc37c92a45b780d91fcda4c85](1111111.assets/67835edc37c92a45b780d91fcda4c85.png)

#### 4. GM(1,N)

##### 3.1 模型名称

一阶多变量灰色预测模型 （One Order Multiple Variable Grey Model, GM(1,N)）

  (1,N) : 一阶微分方程，多个变量

不仅考虑了系统主变量，还考虑若干影响主变量的影响因素

##### 3.2 求解步骤(与GM(1,1)类似)

设系统主变量原始数据序列为
$$
X_1^{(0)}=(x_1^{(0)}(1),x_1^{(0)}(2),\cdots,x_1^{(0)}(m))
$$
相关影响因素序列为
$$
X_i^{(0)}=(x_i^{(0)}(1),x_i^{(0)}(2),\cdots,x_i^{(0)}(m)), \quad i=2,\, \cdots,N
$$
序列
$$
X_i^{(1)}=(x_i^{(0)}(1),x_i^{(0)}(2),\cdots,x_i^{(0)}(m)), \quad i=1,\,2,\,\cdots,N
$$
被称为$X_i^{(0)}(i=1,\,2\,\cdots,N)$的依次累加生成序列，其中
$$
x_i^{(1)}(k)=\sum_{i=1}^{k}{x_i^{(0)}(i)}
$$
序列
$$
Z_1^{(1)}=(-,z_1^{(1)}(2),z_1^{(1)}(3),\cdots,z_1^{(1)}(m))
$$
被称为$X_1^{(1)}$的均值生成序列或系统的背景值，其中
$$
z_1^{(1)}(k)=\frac{1}{2}(x_1^{(1)}(k)+x_1^{(1)}(k-1)),\quad k=2,\,3,\,\cdots,m
$$
则称
$$
x_1^{(0)}(k)+az_1^{(1)}=\sum_{i=2}^{N}{b_ix_i^{(1)}(k)}
$$
为一阶多变量灰色预测模型（One Order Multiple Variable Grey Model,GM(1,N)模型）。

其中参数a是**主变量参数**或系统发展系数，

$b_2,\,b_3,\,\cdots,b_N$是GM(1,N)模型的**灰作用系数**或背景值。

类似地，参数值$\hat{a}=[a,b_2,b_3,\cdots,b_N]^T$可以通过最小二乘方法求得，即设
$$
Y=\left[ \begin{matrix} x_1^{(0)}(2)\\ x_1^{(0)}(3)\\ \vdots\\ x_1^{(0)}(m)\\ \end{matrix} \right]
$$

$$
B=\left[ \begin{matrix} -z_1^{(1)}(2) & x_2^{(1)}(2) & \cdots &  x_N^{(1)}(2) \\ -z_1^{(1)}(3) & x_2^{(1)}(3) & \cdots &  x_N^{(1)}(3)\\ \vdots & \vdots & \ddots & \vdots \\  -z_1^{(1)}(m) & x_2^{(1)}(m) & \cdots & x_N^{(1)}(m)\\  \end{matrix} \right]
$$

则
$$
\hat{a}=(B^TB)^{-1}B^TY
$$
方程
$$
\frac{dx^{(1)}}{dt}+ax_1^{(1)}=b_2x_2^{(1)}+b_3x_3^{(1)}+\cdots+b_Nx_N^{(1)}
$$
被称为GM(1,N)模型的白化方程，也叫影子方程。

结合$\hat{a}=(B^TB)^{-1}B^TY$所求解的参数值，代入上式可以得到
$$
x^{(1)}(t)=e^{-at}\left[\sum_{i=2}^{N}\int{b_ix_i^{(1)}(t)e^{at}dt}+x^{(1)}(0)-\sum_{i=2}^{N}\int{b_ix_i^{(1)}(0)dt}\right]\\  =e^{-at}\left[x_1^{(1)}(0)-t\sum_{i=2}^{N}b_ix_i^{(1)}(0)+\sum_{i=2}^{N}\int{b_ix_i^{(1)}(t)e^{at}dt}\right]
$$
当序列$X_i^{(1)}(i=1,\,2,\,\cdots,N)$的变化是平滑的，$\sum_{i=2}^{N}{b_ix_i^{(1)}(k)}$被视为一个常量的值。可得GM(1,N)模型的时间响应序列为
$$
\hat{x}_1^{(1)}(k+1)=\left(x_1^{(1)}(1)-\frac{1}{a}\sum_{i=2}^{N}b_ix_i^{(1)}(k+1)\right)e^{-ak}+\frac{1}{a}\sum_{i=2}^{N}b_ix_i^{(1)}(k+1),\quad k=1,\,2,\,\cdots,n-1
$$
类似的，通过上式可以求解依次累加生成序列$X_1^{(1)}$的模拟值$\hat{X}_1^{(1)}$，
$$
\hat{X}_1^{(1)}=(\hat{x}_1^{(1)}(1),\hat{x}_1^{(2)}(2),\cdots,\hat{x}_1^{(m)}(m))
$$
运用方程
$$
\hat{x}_1^{(0)}(k+1)=\hat{x}_1^{(1)}(k+1)-\hat{x}_1^{(1)}(k)
$$
可分别求解原始序列的模拟值和预测值$\hat{x}_1^{(0)}(k)$，排列所有$\hat{x}_1^{(0)}(k)$的值可得原始序列的模拟值序列
$$
\hat{X}_1^{(0)}=(\hat{x}_1^{(0)}(1),\hat{x}_1^{(0)}(2),\cdots,\hat{x}_1^{(0)}(m))
$$
类似的，结合$X_1^{(0)}=(x_1^{(0)}(1),x_1^{(0)}(2),\cdots,x_1^{(0)}(m))$和$\hat{X}_1^{(0)}=(\hat{x}_1^{(0)}(1),\hat{x}_1^{(0)}(2),\cdots,\hat{x}_1^{(0)}(m))$可以计算MAPE，得出模型仿真效果，MAPE被定义为
$$
MAPE(\%)=\frac{1}{n}\sum_{k=1}^{n}|\frac{x^{(0)}(k)-\hat{x}^{(0)}(k)}{x^{(0)}(k)}|
$$
不同于GM(1,1)模型的是，更多的变量考虑进入GM(1,N)模型，而不再是单一的系统主变量。

从输入信息和输出结果来看GM(1,N)模型与其它时变多变量模型相似，区别点与GM(1,1)模型类似,GM(1,N)模型建模是基于累加生成序列的，**随机扰动的影响会被降低且系统平稳性更强**。

#### 5. 代码实现

##### 5.1 灰色预测在matlab的实现 示例一

~~~matlab
clc,clear;  
syms a b;  
c=[a b]';  
A=[89677,99215,109655,120333,135823,159878,182321,209407,246619,300670];  
B=cumsum(A);  %原始数据累加  
n=length(A);  
for i=1:(n-1)  
    C(i)=(B(i)+B(i+1))/2; %生成累加矩阵  
end  
%计算待定参数的值  
D=A;D(1)=[];  
D=D';  
E=[-C;ones(1,n-1)];  
c=inv(E*E')*E*D;  
c=c';  
a=c(1);b=c(2);  
%预测后续数据  
F=[];F(1)=A(1);  
for i=2:(n+10)  %只推测后10个数据，可以从此修改  
    F(i)=(A(1)-b/a)/exp(a*(i-1))+b/a;  
end  
G=[];G(1)=A(1);  
for i=2:(n+10)  %只推测后10个数据，可以从此修改  
    G(i)=F(i)-F(i-1);  %得到预测出来的数据  
end  
t1=1999:2008;  
t2=1999:2018;  %多10组数据  
G  
h=plot(t1,A,'o',t2,G,'-'); %原始数据与预测数据的比较  
set(h,'LineWidth',1.5);      

~~~

##### 5.2 灰色预测在matlab的实现 示例二 

~~~matlab
clear

% load DataFile.mat %如需外部调入数据，请首先将数据导入mat文件
% X = Data;
X = [174 179 183 189 207 234 220.5 256 270 285]; %已有观测数据。请根据具体问题修改
X1 = cumsum(X);  % 原始数据累加
n = length(X);
for i=1:(n-1)
    Z(i)=(X1(i)+X1(i+1))/2;  % 计算邻均值数列
end
% 计算数据向量Y和数据矩阵B
Y = X; Y(1) = []; Y = Y';
B=[-Z;ones(1,n-1)];
u = inv(B * B') * B * Y;
u = u';
a = u(1); b = u(2);

% 预测后续数据
X1_GM = []; X1_GM(1) = X(1);
for i = 2:(n + 10)
    X1_GM(i) = (X(1) - b/a) / exp(a*(i-1)) + b/a ;
end
X_GM = []; X_GM(1) = X(1);
for i = 2:(n+10)
    X_GM(i) = X1_GM(i) - X1_GM(i-1); %累减，得到预测数据
end 

% 数据可视化
t1 = 1995:2004;
t2 = 1995:2014;
X_GM, a, b
bar(t1, X, 'r'); %观测数据的柱状图
hold on
plot(t2, X_GM, 'LineWidth', 1); %预测数据曲线图
set(gca,'xtick',[1995:1:2014]);
xlabel('年份'); ylabel('污水量/亿吨')
legend('实际值', '预测值')

% 预测值检验
delta = abs(X - X_GM(1:n));
error = delta / X(1);
S1 = var(X); S2 = var(error);
C = S2 / S1;
P = length(find(delta < 0.6745*S1)) / n;
C, P
~~~

##### 5.3  灰色预测在python中的实现

说明

运行环境为python2.7版本interpreter 需要借助numpy库 数据序列请根据问题需要进行修改 预测序列长度在main函数中进行修改，代码备注处有标明。

~~~python
# -*- coding: utf-8 -*-
 
import numpy as np
import math
 
history_data = [724.57,746.62,778.27,800.8,827.75,871.1,912.37,954.28,995.01,1037.2]
#数据请根据问题需要进行修改
n = len(history_data)
X0 = np.array(history_data)
 
#累加生成
history_data_agg = [sum(history_data[0:i+1]) for i in range(n)]
X1 = np.array(history_data_agg)
 
#计算数据矩阵B和数据向量Y
B = np.zeros([n-1,2])
Y = np.zeros([n-1,1])
for i in range(0,n-1):
    B[i][0] = -0.5*(X1[i] + X1[i+1])
    B[i][1] = 1
    Y[i][0] = X0[i+1]
 
#计算GM(1,1)微分方程的参数a和u
#A = np.zeros([2,1])
A = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
a = A[0][0]
u = A[1][0]
 
#建立灰色预测模型
XX0 = np.zeros(n)
XX0[0] = X0[0]
for i in range(1,n):
    XX0[i] = (X0[0] - u/a)*(1-math.exp(a))*math.exp(-a*(i))
 
 
#模型精度的后验差检验
e = 0      #求残差平均值
for i in range(0,n):
    e += (X0[i] - XX0[i])
e /= n
 
#求历史数据平均值
aver = 0
for i in range(0,n):
    aver += X0[i]
aver /= n
 
#求历史数据方差
s12 = 0
for i in range(0,n):
    s12 += (X0[i]-aver)**2
s12 /= n
 
#求残差方差
s22 = 0
for i in range(0,n):
    s22 += ((X0[i] - XX0[i]) - e)**2
s22 /= n
 
#求后验差比值
C = s22 / s12
 
#求小误差概率
cout = 0
for i in range(0,n):
    if abs((X0[i] - XX0[i]) - e) < 0.6754*math.sqrt(s12):
        cout = cout+1
    else:
        cout = cout
P = cout / n
 
if (C < 0.35 and P > 0.95):
    #预测精度为一级
    m = 10   #请输入需要预测的年数
    print('往后m各年负荷为：')
    f = np.zeros(m)
    for i in range(0,m):
        f[i] = (X0[0] - u/a)*(1-math.exp(a))*math.exp(-a*(i+n))
        print f[i]
else:
    print('灰色预测法不适用')
~~~

##### 5.4 灰色预测在C++中的实现

https://pan.baidu.com/s/1FxLLLZtNdOAs8zLC7To5gw#list/path=%2Fsharelink2785985965-169505872955451%2F%E7%81%B0%E8%89%B2%E9%A2%84%E6%B5%8BC%2B%2B&parentPath=%2Fsharelink2785985965-169505872955451

提取码：mspx

#### 6. 参考资料

1. [数学建模培训营----灰色预测](https://anl.sjtu.edu.cn/mcm/docs/1%E6%A8%A1%E5%9E%8B/7%E9%A2%84%E6%B5%8B%E4%B8%BB%E9%A2%98/1%E7%A6%BB%E6%95%A3%E5%9E%8B%E9%A2%84%E6%B5%8B/%E7%81%B0%E8%89%B2%E9%A2%84%E6%B5%8B/doc)
2. [B站----灰色预测]( https://b23.tv/iqRHxp)

