% 求解pi
clc, clear
sim = monte_carlo(100000, @gen, @cond) * 4.0;
disp('结果：');
disp(sim);
function Point = gen()
    Point = rand([1,2]);
end
function ans = cond(x)
    ans = sqrt((x(1)-0.5).^2+(x(2)-0.5).^2) < 0.5;
end