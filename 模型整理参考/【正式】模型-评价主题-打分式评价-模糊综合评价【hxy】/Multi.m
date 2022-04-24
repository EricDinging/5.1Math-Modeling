clc, clear
% Input original data
r = [0.8 0.15 0.05 0 0
0.2	0.6	0.1	0.1	0
0.5	0.4	0.1	0 0
0.1	0.3	0.5	0.05 0.05
0.3	0.5	0.15 0.05 0
0.2	0.2	0.4	0.1	0.1
0.4	0.4	0.1	0.1	0
0.1	0.3	0.3	0.2	0.1
0.3	0.2	0.2	0.2	0.1
0.1	0.3	0.5	0.1	0
0.2	0.3	0.3	0.1	0.1
0.2	0.3	0.35 0.15 0
0.1	0.3	0.4	0.1	0.1
0.1	0.4	0.3	0.1	0.1
0.3	0.4	0.2	0.1	0
0.1	0.4	0.3	0.1	0.1
0.2	0.3	0.4	0.1	0
0.4	0.3	0.2	0.1	0];
% Set the First order weighted allocation A
A = [0.4 0.3 0.2 0.1];
% Set the Second order weighted allocation A1 to A4
A1 = [0.2 0.3 0.3 0.2];
A2 = [0.3 0.2 0.1 0.2 0.2];
A3 = [0.1 0.2 0.3 0.2 0.2];
A4 = [0.3 0.2 0.2 0.3];
% Calculate the Second order evaluation vector B1 to B4
R(1,:) = A1 * r((1:4),:);
R(2,:) = A2 * r((5:9),:);
R(3,:) = A3 * r((10:14),:);
R(4,:) = A4 * r((15:end),:);
% Calculate the First order evaluation vector B
B = A * R;

