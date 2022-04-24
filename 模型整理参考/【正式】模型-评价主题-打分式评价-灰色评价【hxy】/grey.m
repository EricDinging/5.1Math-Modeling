clc, clear
data = [1064.0 21.6 1300.0 1001.1 3880.0 15.0 5.0;
				1061.0 22.0 1349.0 1074.0 4016.0 12.8 5.0;
				1015.0 22.0 1402.0 968.0  4022.0 11.2 4.8;
				1125.0 23.0 1234.0 1010.0 4362.0 11.0 4.7];
% 数据预处理
% 求每一列的均值
Mean = mean(data);
% 每个元素除以均值,repmat扩充均值和data一样大
new_data = data ./ repmat(Mean,size(data,1),1);
disp('预处理后的数据矩阵为：');
disp(new_data);
% O为理想值
O = new_data(1,:);
% A为实际值
A = new_data(2:4,:);
% 计算|X0-Xi|
Abs = abs(A - repmat(O,size(A,1),1));
% 计算两级最小值
Min = min(min(Abs));
% 计算两级最大值
Max = max(max(Abs));
% 计算评价矩阵R
theta = 0.5;
R = (Min + theta * Max)./(Abs + theta * Max);
disp('评价矩阵R为：');
disp(R);
% 设置权重矩阵W
W = [0.3500 0.1500 0.1000 0.1000 0.1000 0.1500 0.0500];
% 计算灰色关联矩阵A
A = W * R';
disp('灰色关联矩阵为：');
disp(A);
