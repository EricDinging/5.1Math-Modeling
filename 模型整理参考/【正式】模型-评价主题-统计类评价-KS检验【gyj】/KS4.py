#KS检验
from math import *
#由于积分是用步长计算的，存在一定误差，推荐使用matlab版本或使用SPSS软件
#需要输入样本，即为X，检查X是否符合正态分布，会输出maxD，查D的参数表，当D<D(N,α)时（N为样本容量，α为显著性水平），分布相同。反之，则分布不同
#使用python3.6

if __name__ == '__main__':
    #X为待检验的样本，检验其是否服从正态分布
    X=[87,77,92,68,80,78,84,77,81,80,80,77,92,86,76,80,81,75,77,72,81,72,84,86,80,68,77,87,76,77,78,92,75,80,78]
    N=len(X)
    #下面为计算样本不同情况的累积频率
    X.sort()#对X排序
    #下面为将X中重复的数字去掉并确定每一种数字的数目
    tmp1=[]#tmp1为将X中重复数字去掉并排序的序列
    tmp2=[]#tmp2为tmp1中对应数字的频数
    num=1
    for i in range(N-1):
        if(X[i]!=X[i+1]):
            tmp1.append(X[i])
            tmp2.append(num)
            num=1
        else:
            num=num+1
    tmp1.append(X[N-1])
    tmp2.append(num)
    #下面为计算样本不同情况的累积频率
    Fn=[]
    F=0#初始化累计次数
    for i in range(len(tmp1)):
        F +=tmp2[i]
        Fn.append(F/N)
    #下面为求解样本拟合的正态分布
    #mu为期望，sigma为标准差
    mu=sum(X)/N
    sigma=0
    for i in range(N):
        sigma +=(X[i]-mu)*(X[i]-mu)
    sigma=sqrt(sigma/(N-1))
    #用积分计算样本所拟合的正态分布的理论累积频率
    #由于积分是用步长计算的，存在一定误差
    A=[]
    p=0#初始化累计频率
    l=min(min(tmp1),mu-10*sigma)#计算积分从该值来替代负无穷大
    for i in range(len(tmp1)):
        while l<tmp1[i]:
            p +=1/(sigma*sqrt(2*pi))*exp(-(l-mu)*(l-mu)/(2*sigma*sigma))*sigma/1000
            l +=sigma/1000
        A.append(p)
    #得到样本与理论分布函数差
    D=[]
    for i in range(len(tmp1)):
        D.append(abs(Fn[i]-A[i]))
    maxD=max(D)
    print("样本与理论分布函数差的最大值为:")
    print(maxD)
    #查D的参数表，当D<D(N,α)时（N为样本容量，α为显著性水平），分布相同
    #反之，则分布不同
    print("查D的参数表，当D<D(N,α)时（N为样本容量，α为显著性水平），分布相同.反之，则分布不同")
