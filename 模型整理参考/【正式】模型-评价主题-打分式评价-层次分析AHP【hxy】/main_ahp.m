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