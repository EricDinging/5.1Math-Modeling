% 求解积分
clc, clear
sim = monte_carlo(100000, @gen, @cond);
disp('结果：');
disp(sim);
function Point = gen()
    Point = rand([1,2]);
end
function ans = cond(x)
    ans = x(2) < x(1)^2;
end