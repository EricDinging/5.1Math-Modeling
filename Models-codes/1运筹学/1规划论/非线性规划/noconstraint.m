%% symbolic
clc, clear;
syms x y 
f = x^3 - y^3 + 3 * x^2 + 3 * y^2 - 9 * x;
df = jacobian(f);
d2f = jacobian(df);
[xx, yy] = solve(df)
xx = double(xx); yy = double(yy);
for i = 1:length(xx)
    a = subs(d2f, {x, y}, {xx(i), yy(i)});
    b = eig(a);
    f = subs(f, {x, y}, {xx(i), yy(i)});
    if all(b > 0)
        fprintf('(%f, %f)是极小值点，极小值是%f \n', xx(i), yy(i), f);
    elseif all(b < 0)
        fprintf('(%f, %f)是极大值点，极大值是%f \n', xx(i), yy(i), f);
    elseif any(b > 0) & any(b < 0)
        fprintf('(%f, %f)不是极值点\n', xx(i), yy(i));
    else
        fprintf('无法判断(%f, %f)\n', xx(i), yy(i));
    end
end

