f = [-2;-3;5]; %min
a = [-2, 5, -1; 1, 3, 1];
b = [-10; 12];
aeq = [1, 1, 1];
beq = 7;
[x, y] = linprog(f, a, b, aeq, beq, zeros(3,1));
x,y = -y;

%% 投资的收益和风险

r = [5, 28, 21, 23, 25] * 0.01;
q = [0, 2.5, 1.5, 5.5, 2.6] * 0.01;
p = [0, 1, 2, 4.5, 6.5] * 0.01;
M = 1;
for a = 0:0.001:0.05
    f = -(r - p)';
    A = diag(q/M);
    b = a * ones(5, 1);
    Aeq = 1 + p;
    beq = M;
    lb = zeros(5,1);
    [x, Q] = linprog(f, A, b, Aeq, beq, lb);
    Q = -Q;
    plot(a, Q, '*k');
    hold on;
end 
xlabel('a');
ylabel('Q');






