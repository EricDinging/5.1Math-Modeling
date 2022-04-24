clc, clear
% Input original data
R = [0.1 0.5 0.4 0 0
0.2	0.5	0.2	0.1 0
0.2	0.5	0.3	0 0
0.2	0.6	0.2	0 0];
% Set the weighted allocation A
A = [0.25 0.2 0.25 0.3];
% Calculate the evaluation vector B
B = A * R;
