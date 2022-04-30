import numpy as np
y=np.array([6.35,6.20,6.22,6.66,7.15,7.89,8.72,8.94,9.28,9.80])
def WeightMoveAverage(y,N):
    Mt=['*']*N
    for i in range(N,len(y)+1):
        M=0
        Sum=0
        for j in range(N,0,-1):
            M+=j*y[i-N+j-1]
            Sum+=j
        Mt.append(M/Sum)
    return Mt
yt3=WeightMoveAverage(y,3) 
s3=np.sqrt(((y[3:]-yt3[3:-1])**2).mean())
print('N=3时,预测值：',yt3,'，预测的标准误差：',s3)
