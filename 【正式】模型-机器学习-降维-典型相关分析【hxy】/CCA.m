clc, clear

% X，Y行数为样本数量，列数为属性数
X = readmatrix("Book1.xlsx","Sheet","Sheet2");
Y = readmatrix("Book1.xlsx","Sheet","Sheet1");
% 设定显著性水平
a = 0.05;

% p为X的属性数，q为Y的属性数，n为样本数
p = size(X,2);
q = size(Y,2);
n = size(X,1);
% 标准化矩阵X，Y
X = zscore(X);
Y = zscore(Y);
% 得到相关系数矩阵R
C = [X Y];
R = corrcoef(C);
R11 = R((1:p),(1:p));
R12 = R((1:p),(p+1:end));
R21 = R12';
R22 = R((p+1:end),(p+1:end));
disp('相关系数矩阵:');
disp(R);

% A1和B1为典型变量系数，r1为典型相关系数
% U1和V1为典型变量的值，stats为假设检验的统计量的值
[A1,B1,r1,U1,V1,stats] = canoncorr(X,Y);
% 修正正负号，使每一列的系数和为正
A1 = A1 .* repmat(sign(sum(A1)),size(A1,1),1);
B1 = B1 .* repmat(sign(sum(B1)),size(B1,1),1);
U1 = U1 .* repmat(sign(sum(A1)),size(U1,1),1);
V1 = V1 .* repmat(sign(sum(B1)),size(V1,1),1);

% 输出统计数据
disp('CCA统计数据:');
disp(stats);
% 得到显著性检验后被选取的典型变量数量
count = 0;
for i=1:length(stats.pF)
    if(stats.pF(i) < a)
        count = count + 1;
    end
end
% 选取经过显著性检验的典型变量
A = A1(:,1:count);
B = B1(:,1:count);
r = r1(:,1:count);
U = U1(:,1:count);
V = V1(:,1:count);

disp(' ');
disp('---------------------- 分割线 ----------------------');
disp(' ');
disp('完整数据');
disp(' ');
disp('X典型变量系数矩阵A‘：')
disp(A1);
disp('Y典型变量系数矩阵B’：')
disp(B1);
disp('典型相关系数矩阵r‘：')
disp(r1);
disp('X典型变量值矩阵U‘：')
disp(U1);
disp('Y典型变量值矩阵V‘：')
disp(V1);

% 计算R(X,U'), R(Y,V')
X_U_R1 = R11 * A1;
Y_V_R1 = R22 * B1;
disp('原始变量X与本组典型变量U之间的相关系数：');
disp(X_U_R1);
disp('原始变量Y与本组典型变量V之间的相关系数：');
disp(Y_V_R1);
% 计算R(X,V'), R(Y,U')
X_V_R1 = R12 * B1;
Y_U_R1 = R21 * A1;
disp('原始变量X与对应组典型变量V之间的相关系数：');
disp(X_V_R1);
disp('原始变量Y与对应组典型变量U之间的相关系数：');
disp(Y_U_R1);

% 计算方差比例与累计方差比例
% x～u
ux = sum(X_U_R1 .^2) / p;
disp('X组原始变量被u_i解释的方差比例：');
disp(ux);
ux_cum = cumsum(ux);
disp('X组原始变量被u_i解释的方差累计比例：');
disp(ux_cum);
% x～v
vx = sum(X_V_R1 .^2) / p;
disp('X组原始变量被v_i解释的方差比例：');
disp(vx);
vx_cum = cumsum(vx);
disp('X组原始变量被v_i解释的方差累计比例：');
disp(vx_cum);
% y～v
vy = sum(Y_V_R1 .^2) / q;
disp('Y组原始变量被v_i解释的方差比例：');
disp(vy);
vy_cum = cumsum(vy);
disp('Y组原始变量被v_i解释的方差累计比例：');
disp(vy_cum);
% y～u
uy = sum(Y_U_R1 .^2) / q;
disp('Y组原始变量被u_i解释的方差比例：');
disp(uy);
uy_cum = cumsum(uy);
disp('Y组原始变量被u_i解释的方差累计比例：');
disp(uy_cum);
disp(' ');
disp('---------------------- 分割线 ----------------------');
disp(' ');
disp('显著性检验后数据');
disp(' ');
disp('X典型变量系数矩阵A：')
disp(A);
disp('Y典型变量系数矩阵B：')
disp(B);
disp('典型相关系数矩阵r：')
disp(r);
disp('X典型变量值矩阵U：')
disp(U);
disp('Y典型变量值矩阵V：')
disp(V);

% 计算R(X,U), R(Y,V)
X_U_R = R11 * A;
Y_V_R = R22 * B;
disp('原始变量X与本组典型变量U之间的相关系数：');
disp(X_U_R);
disp('原始变量Y与本组典型变量V之间的相关系数：');
disp(Y_V_R);
% 计算R(X,V), R(Y,U)
X_V_R = R12 * B;
Y_U_R = R21 * A;
disp('原始变量X与对应组典型变量V之间的相关系数：');
disp(X_V_R);
disp('原始变量Y与对应组典型变量U之间的相关系数：');
disp(Y_U_R);

% 计算方差比例与累计方差比例
% x～u
ux = sum(X_U_R .^2) / p;
disp('X组原始变量被u_i解释的方差比例：');
disp(ux);
ux_cum = cumsum(ux);
disp('X组原始变量被u_i解释的方差累计比例：');
disp(ux_cum);
% x～v
vx = sum(X_V_R .^2) / p;
disp('X组原始变量被v_i解释的方差比例：');
disp(vx);
vx_cum = cumsum(vx);
disp('X组原始变量被v_i解释的方差累计比例：');
disp(vx_cum);
% y～v
vy = sum(Y_V_R .^2) / q;
disp('Y组原始变量被v_i解释的方差比例：');
disp(vy);
vy_cum = cumsum(vy);
disp('Y组原始变量被v_i解释的方差累计比例：');
disp(vy_cum);
% y～u
uy = sum(Y_U_R .^2) / q;
disp('Y组原始变量被u_i解释的方差比例：');
disp(uy);
uy_cum = cumsum(uy);
disp('Y组原始变量被u_i解释的方差累计比例：');
disp(uy_cum);
disp('---------------------- END ----------------------');
