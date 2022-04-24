clear
X=[ 60 70 85 106 35;
156 200 157 263 105;
50 180 100 86 30];        %用户输入多指标输入矩阵X
Y=[ 80 60 90 96 30;
12 13 20 17 8;
27 25 15 28 3;
4 2 5 5 1];        %用户输入多指标输出矩阵Y
n=size(X',1);m=size(X,1);s=size(Y,1);
epsilon=10^-10;     
%定义非阿基米德无穷小 =10^(－10)
f=[zeros(1,n) -epsilon*ones(1,m+s) 1];
A=zeros(1,n+m+s+1); b=0;
LB=zeros(n+m+s+1,1);UB=[ ];
LB(n+m+s+1)=-Inf;
for i=1:n;
Aeq=[X  eye(m)     zeros(m,s)   -X(:,i)
Y  zeros(s,m)   -eye(s)      zeros(s,1)];
beq=[zeros(m,1)
Y(:,i)];
w(:,i)= linprog (f,A,b,Aeq,beq,LB,UB);   
%解线性规划，得DMUi的最佳权向量wi
end
w                             %输出最佳权向量
lambda=w(1:n,:)          %输出 lambda*
s_minus=w(n+1:n+m,:)     %输出s*－
s_plus=w(n+m+1:n+m+s,:)  %输出s*＋
theta=w(n+m+s+1,:)        %输出 theta*