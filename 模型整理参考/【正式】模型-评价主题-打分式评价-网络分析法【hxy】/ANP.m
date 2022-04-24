clc
clear
close all

%选车算例的解答
%% 列出所有的判断矩阵
%成本指标下美日欧三种车间的判断矩阵
A1=[1,5,3;
  1/5,1,1/3;
  1/3,3,1];
%维修指标下美日欧三种车间的判断矩阵
A2=[1,5,2;
  1/5,1,1/3;
  1/2,3,1];
%耐用性指标下美日欧三种车间的判断矩阵
A3=[1,1/5,1/3;
  5,1,3;
  3,1/3,1];
%美车三种指标下间的判断矩阵
A4=[1,3,4;
  1/3,1,1;
  1/4,1,1];
%欧车三种指标下间的判断矩阵
A5=[1,1,1/2;
  1,1,1/2;
  2,2,1];
%日车三种指标下间的判断矩阵
A6=[1,2,1;
  1/2,1,1/2;
  1,2,1];

%% 计算6个判断矩阵的特征向量
%计算6个判断矩阵的特征向量和CR
[B1,D1]=eig(A1);
[B2,D2]=eig(A2);
[B3,D3]=eig(A3);
[B4,D4]=eig(A4);
%严格得说每个判断矩阵都需要判断是不是一致阵，如果是的话就不需要再求特征向量和特征根
%即：
%由于rank(A5)=rank(A6)=1是一致阵，故下面两步不再需要
%[B5,D5]=eig(A5);
%[B6,D6]=eig(A6);
%这里作者是算出来答案跟结果不对才回去判断的一致阵，所以程序编得不是很严谨，从逻辑上来说是错误的。即看着结果推程序，所以很遗憾本程序对于解答的推广无积极意义。
%希望读者自行修改程序，先判断是否是一致阵，再计算特征根特征向量
%直接将各自的第一列设为特征向量
B5=A5(1:3,1);
B6=A6(1:3,1);

%求λmax最大特征根
lamdaMax1=max(max(D1));
lamdaMax2=max(max(D2));
lamdaMax3=max(max(D3));
lamdaMax4=max(max(D4));
%一致阵的最大特征根为阶数n
lamdaMax5=length(A5);
lamdaMax6=length(A6);

n=length(A1);
CI(1)=(lamdaMax1-n)/(n-1);
CI(2)=(lamdaMax2-n)/(n-1);
CI(3)=(lamdaMax3-n)/(n-1);
CI(4)=(lamdaMax4-n)/(n-1);
CI(5)=(lamdaMax5-n)/(n-1);
CI(6)=(lamdaMax6-n)/(n-1);
%CI=(λmax-n)/(n-1)
%对于1-9阶判断矩阵，RI值如下：
%1   2   3     4    5    6    7   8    9
%0   0  0.58 0.90 1.12 1.24 1.32 1.41 1.45
%由于n=3故RI（3）=0.58
RI=0.58;
%判断矩阵的一致性指标CI与同阶平均随机一致性指标RI之比称为随机一致性比率CR，当CR=CI/RI<0.10时，可以认为判断矩阵具有满意的一致性，否则需要调整判断矩阵。
%CR=CI/RI=0.0285/1.12=0.0255<0.10 因此，通过一致性检验。
for i=1:6
    CR(i)=CI(i)/RI;
    if CR(i)<0.1
        disp(strcat('判断矩阵',num2str(i),'通过一致性检验'));
    else
        disp(strcat('判断矩阵',num2str(i),'未通过一致性检验'));
    end
end

%权重即为最大特征根对应的特征向量W进行归一化后的结果，w=W./sum(W)
W1=B1(1:n,1);
w1=W1./sum(W1);

W2=B2(1:n,1);
w2=W2./sum(W2);

W3=B3(1:n,1);
w3=W3./sum(W3);

W4=B4(1:n,1);
w4=W4./sum(W4);

W5=B5(1:n,1);
w5=W5./sum(W5);

W6=B6(1:n,1);
w6=W6./sum(W6);
%% 输出阶段结果
CR
w1
w2
w3
w4
w5
w6
%% 计算极限超矩阵
%再考虑成本、维修和耐用性之间的相互影响，得到三者的权重矩阵
C=[0.3,0.2,0.6;
    0.4 0.25 0.3;
    0.3 0.55 0.1];
%初始超矩阵
%行列均为：成本 维修 耐用性 美 欧 日
super=[C  w4 w5  w6
       w1 w2 w3 zeros(3)];
%令归一化的排序向量A为[0.5,1;0.5,0],则加权超矩阵为[0.5*WW1 1*WW2;0.5*WW3 0*WW4]
%A=[ones(6,3)*0.5 ones(6,3)];appears to be a bad example
WW1=super(1:3,1:3);
WW2=super(1:3,4:6);
WW3=super(4:6,1:3);
WW4=super(4:6,4:6);
%powerSuper=A.*super;appears to be a bad example
powerSuper=[0.5*WW1 1*WW2;0.5*WW3 0*WW4];
ppowerSuper=powerSuper;
for m= 1:6
    P1=ppowerSuper(1:6,1);
    P2=ppowerSuper(1:6,2);
    P3=ppowerSuper(1:6,3);
    P4=ppowerSuper(1:6,4);
    P5=ppowerSuper(1:6,5);
    P6=ppowerSuper(1:6,6);
    p1=P1./sum(P1);
    p2=P2./sum(P2);
    p3=P3./sum(P3);
    p4=P4./sum(P4);
    p5=P5./sum(P5);
    p6=P6./sum(P6);
    ppowerSuper=[p1,p2,p3,p4,p5,p6]^2;
end
%% 输出最终结果
ppowerSuper