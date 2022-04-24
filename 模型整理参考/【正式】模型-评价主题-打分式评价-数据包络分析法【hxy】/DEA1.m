clear
X=[ 60 70 85 106 35;
156 200 157 263 105;
50 180 100 86 30];        %用户输入多指标输入矩阵X
Y=[ 80 60 90 96 30;
12 13 20 17 8;
27 25 15 28 3;
4 2 5 5 1];        %用户输入多指标输出矩阵Y
n=size(X',1);m=size(X,1);s=size(Y,1);
A=[-X'   Y'];
b=zeros(n,1);
LB=zeros(m+s,1);UB=[ ];
for i=1:n;
f=[zeros(1,m) -Y(:,i)'];
Aeq=[X(:,i)' zeros(1,s)];beq=1;
w(:,i)=linprog(f,A,b,Aeq,beq,LB,UB);    
%解线性规划，得DMUi的最佳权向量wi
E(i, i)=Y(:,i)'*w(m+1:m+s,i);            
%求出DMUi的相对效率值Eii
end
w                 %输出最佳权向量
E                 %输出相对效率值Eii
omega=w(1:m,:)     %输出投入权向量omega
mu=w(m+1:m+s,:)   %输出产出权向量mu 