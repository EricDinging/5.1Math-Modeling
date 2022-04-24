% 求解曲边三角形面积， 蒙特卡洛
clc, clear;
x = unifrnd(0, 12, [1,100000000]);
y = unifrnd(0, 9, [1,100000000]);
f = (sum(y < x.^2 &x <= 3) + sum(y<12-x & x>3))/length(x);
area = 12 * 9 * f;

%%
% 非线性，蒙特卡洛
clc, clear;
rand('state', sum(clock));
p0 = 0;
tic
for i = 1:10^6
    X = randi([0, 99], 1, 5);
    [f, g] = mente(X);
    if all(g <= 0)
        if p0 < f
            X0 = X;
            p0 = f;
        end
    end
end
toc

%%
% 线性整数规划
clc, clear;
c = [3, 8, 2, 10, 3; 8, 7, 2, 9, 7; 6, 4, 2, 7, 5; 8, 4, 2, 3, 5; 9, 10, 6, 9, 10];
c = c(:);
a = zeros(10, 25);
intcon = 1:25;
for i = 1:5
    a(i, (i - 1) * 5 + 1:5 * i) = 1;
    a(i + 5, i:5:25) = 1;
end
b = ones(10, 1);
lb = zeros(25, 1);
ub = ones(25, 1);
[x, f] = intlinprog(c, intcon, [], [], a, b, lb, ub);
x = reshape(x, [5, 5]);

%%

function [f, g] = mente(x)
    f = x(1)^2 + x(2)^3 + 3*x(3)^2 + 4*x(4)^2 + 2 * x(5)^2 - 8*x(1)-2*x(2)-3*x(3)-x(4)-2*x(5);
    g = [sum(x) - 400; x(1)+2*x(2)+2*x(3)+x(4)+6*x(5) - 800;2*x(1)+x(2)+6*x(3)-200; x(3)+x(4)+5*x(5)-200];
end
