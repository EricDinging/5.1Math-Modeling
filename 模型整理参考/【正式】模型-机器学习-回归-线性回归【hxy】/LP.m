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

