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