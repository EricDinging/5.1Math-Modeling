#include <iostream>
#include <math.h>
#include<algorithm>
#define pi 3.1415926
using namespace std;
//由于积分是用步长计算的，存在一定误差，推荐使用matlab版本或使用SPSS软件
//需要输入样本，即为X，检查X是否符合正态分布，会输出maxD，
//查D的参数表，当D<D(N,α)时（N为样本容量，α为显著性水平），分布相同。反之，则分布不同
int main()
{
    //X为待检验的样本，检验其是否服从正态分布
    //N为样本的数目，请先输入
    const int N=35;
    int n=0;//n为样本中不重复的数字的数目，后面将求解
    double X[N]={87,77,92,68,80,78,84,77,81,80,80,77,92,86,76,80,81,75,77,72,81,72,84,86,80,68,77,87,76,77,78,92,75,80,78};
    //下面为计算样本不同情况的累积频率
    sort(X,X+N);//对X排序
    //下面为将X中重复的数字去掉并确定每一种数字的数目
    double tmp1[N],tmp2[N];
    int num=1;
    for(int i=0;i<N-1;i++)
        if(X[i]!=X[i+1])
        {
            tmp1[n]=X[i];
            tmp2[n]=num;
            num=1;
            n++;
        }
        else
            num++;
    tmp1[n]=X[N-1];
    tmp2[n]=num;
    n++;
    //下面为计算样本不同情况的累积频率
    double *Fn;
    Fn=new double[n];
    double F=0;//初始化累计次数
    for(int i=0;i<n;i++)
    {
        F +=tmp2[i];
        Fn[i]=F/N;
    }
    //下面为求解样本拟合的正态分布
    //mu为期望，sigma为标准差
    double mu=0,sigma=0;
    for(int i=0;i<n;i++)
        mu +=tmp1[i]*tmp2[i];
    mu /=N;
    for(int i=0;i<N;i++)
        sigma +=pow((X[i]-mu),2);
    sigma=sqrt(sigma/(N-1));
    //用积分计算样本所拟合的正态分布的理论累积频率
    //由于积分是用步长计算的，存在一定误差
    double *A;
    A=new double[n];
    double p=0,l;//初始化累计频率
    l=tmp1[0]<(mu-10*sigma)?tmp1[0]:(mu-10*sigma);
    for(int i=0;i<n;i++)
    {
        while(l<tmp1[i])
        {
            p +=1/(sigma*sqrt(2*pi))*exp(-(l-mu)*(l-mu)/(2*sigma*sigma))*sigma/1000;
            l +=sigma/1000;
        }
        A[i]=p;
    }
    //得到样本与理论分布函数差
    double *D;
    D=new double[n];
    double maxD=0;
    for(int i=0;i<n;i++)
    {
        D[i]=(Fn[i]-A[i])>0?(Fn[i]-A[i]):(-Fn[i]+A[i]);
        if(D[i]>maxD)
            maxD=D[i];
    }
    cout<<"样本与理论分布函数差的最大值为:"<<maxD<<endl;
    cout<<"查D的参数表，当D<D(N,α)时（N为样本容量，α为显著性水平），分布相同.反之，则分布不同";
    //查D的参数表，当D<D(N,α)时（N为样本容量，α为显著性水平），分布相同.反之，则分布不同
    return 0;
}