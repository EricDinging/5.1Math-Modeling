%KS检验
%注意事项
%需要输入样本，即为X，检查X是否符合正态分布，会输出maxD
%查D的参数表，当D<D(N,α)时（N为样本容量，α为显著性水平），分布相同。反之，则分布不同

%X为待检验的样本，检验其是否服从正态分布
X=[87 77 92 68 80 78 84 77 81 80 80 77 92 86 76 80 81 75 77 72 81 72 84 86 80 68 77 87 76 77 78 92 75 80 78];
N=length(X);
tmp1=unique(X);%将X中重复的样本除去
tmp2=histc(X,tmp1);%确定每一种情况的数目
%下面为计算样本不同情况的累积频率
F=0; %初始化累计次数
Fn=zeros(1,length(tmp1));%初始化数组，Fn为样本每种情况的累计频率
for i=1:length(tmp1)
    F=F+tmp2(i);
    Fn(i)=F/N;
end

%A=zeros(1,length(tmp1));%初始化数组，A为理论每种情况的累计频数
for i=1:length(tmp1)    
    l=min(min(tmp1),mu-10*sigma);
    x=l:sigma/1000:tmp1(i);
    f=1/(sigma*sqrt(2*pi))*exp(-(x-mu).^2/(2*sigma^2));
    A(i)=trapz(x,f);
end
    

    u=mean(X);  
    d=std(X,1);
    A=zeros(1,length(tmp1));
    for i=1:length(tmp1)
        l=min(min(tmp1),u-10*d);
        x=l:d/1000:tmp1(i);
        f=1/(d*sqrt(2*pi))*exp(-(x-u).^2/(2*d^2));
        A(i)=trapz(x,f);
    end


maxD=max(abs(Fn-A))
%end