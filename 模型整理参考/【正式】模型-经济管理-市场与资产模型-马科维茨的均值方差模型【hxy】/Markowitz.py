import pandas as pd
stock=pd.read_table('stock.txt',sep='\t',index_col='Trddt')
stock.index=pd.to_datetime(stock.index)
fjgs = stock.loc[stock.Stkcd==600033,"Dretwd"]
fjgs.name="fjgs"
zndl=stock.loc[stock.Stkcd==600023,'Dretwd']
zndl.name='zndl'
sykj=stock.loc[stock.Stkcd==600183,'Dretwd']
sykj.name='sykj'
hxyh=stock.loc[stock.Stkcd==600015,'Dretwd']
hxyh.name='hxyh'
byjc=stock.loc[stock.Stkcd==600004,'Dretwd']
byjc.name='byjc'
sh_return = pd.concat([fjgs,zndl,sykj,kxyh,byjc],axis=1)
sh_return.head()

# 查看各股的回报率
sh_return=sh_return.dropna()
sh_return.corr()  #删除nan，以及空缺的
sh_return.plot()

# 查看各股的累计收益
cumreturn= (1+sh_return).cumprod()  #(1+sh_return)  +1 表示加上本身
cumreturn.plot()  #累计收益

# 查看各股回报率的相关性
sh_return.corr()

# 计算最优资产比例，绘制最小方差前沿曲线
import ffn
from scipy import linalg
class MeanVariance:
    #定义构造器，传入收益率数据
    def __init__(self,returns):
        self.returns=returns
    #定义最小化方差的函数，即求解二次规划
    def minVar(self,goalRet):
        covs=np.array(self.returns.cov())
        means=np.array(self.returns.mean())
        L1=np.append(np.append(covs.swapaxes(0,1),[means],0),
                     [np.ones(len(means))],0).swapaxes(0,1)
        L2=list(np.ones(len(means)))
        L2.extend([0,0])
        L3=list(means)
        L3.extend([0,0])
        L4=np.array([L2,L3])
        L=np.append(L1,L4,0)
        results=linalg.solve(L,np.append(np.zeros(len(means)),[1,goalRet],0))
        return(np.array([list(self.returns.columns),results[:-2]]))
    #定义绘制最小方差前缘曲线函数
    def frontierCurve(self):
        goals=[x/500000 for x in range(-100,4000)]
        variances=list(map(lambda x: self.calVar(self.minVar(x)[1,:].astype(np.float)),goals))
        plt.plot(variances,goals)
    #给定各资产的比例，计算收益率均值  
    def meanRet(self,fracs):#fracs 概率
        meanRisky=ffn.to_returns(self.returns).mean()
        assert len(meanRisky)==len(fracs), 'Length of fractions must be equal to number of assets'
        return(np.sum(np.multiply(meanRisky,np.array(fracs))))
    #给定各资产的比例，计算收益率方差
    def calVar(self,fracs):
        return(np.dot(np.dot(fracs,self.returns.cov()),fracs))
    # 描绘最小方差前沿曲线
    minvar=MeanVariance(sh_return)
    minvar.frontierCurve() #收益方差最低点

# 选取训练集和测试集
train_set=sh_return["2014"]
test_set=sh_return["2015"]#训练数据，分别导入2014，2015
# 选取组合
varMinmize = MeanVariance(train_set)
goal_return = 0.003
profolio_weight = varMinmize.minVar(goal_return)  #计算最小方差
profolio_weight 
'''
profolio_weight 权重，五种股票的比例权重
array([['fjgs', 'zndl', 'sykj', 'kxyh', 'byjc'],
       ['0.8121632841002377', '-0.4801112026506777',
        '0.43018219629259896', '0.34747305363072123',
        '-0.10970733137288022']], dtype='<U32')
投资权重出现负数，因为可以做空
'''
# 计算测试集的收益率
test_return=np.dot(test_set,np.array([profolio_weight[1,:].astype(np.float)]).swapaxes(0,1))
test_return=pd.DataFrame(test_return,index=test_set.index)
test_cum_return = (1+test_return).cumprod()#累计收益
plt.plot(test_cum_return.index,test_cum_return)

# 与随机生成的组合进行比较
sim_weight=np.random.uniform(0,1,(100,5))
# 叠加
sim_weight=np.apply_along_axis(lambda x: x/sum(x),1,sim_weight)#随机数
sim_return=np.dot(test_set,sim_weight.swapaxes(0,1))
sim_return=pd.DataFrame(sim_return,index=test_cum_return.index)
sim_cum_return=(1+sim_return).cumprod()
plt.plot(sim_return.index,sim_cum_return)
plt.plot(test_cum_return.index,test_cum_return,color="yellow")
