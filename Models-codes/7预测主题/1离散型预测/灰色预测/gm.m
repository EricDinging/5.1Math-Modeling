% GM(1,1)
clc,clear;
x0 = [71,1 72.4 72.4 72.1 71.4 72.0 71.6]';
n = length(x0);
lamda = x0(1:n-1) ./ x0(2:n);
range = minmax(lamda');
x1 = cumsum(x0);
z = (x1(1:n-1) + x1(2:n))/2;
B = [-z, ones(n-1, 1)];
Y = x0(2:n);
u = B \ Y;
syms x(t)
x = dsolve(diff(x) + u(1) * x == u(2), x(0) == x0(1));
y1 = subs(x, t, (0:n-1));
y1 = double(y1);
y = [x0(1), diff(y1)];
epsilon = x0' - y;
delta = abs(epsilon ./ x0');
rho = 1 - (1 - 0.5 * u(1)) / (1 + 0.5 * u(1)) * lamda';


