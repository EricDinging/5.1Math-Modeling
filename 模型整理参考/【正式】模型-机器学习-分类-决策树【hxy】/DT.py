from math import log
import operator

# 计算香农熵(Shannon Entropy)
def calcShannonEnt(dataSet):
    # 计算样本数量
    numEntries=len(dataSet)
    # 字典记录每一个属性(key)和其对应的数量(value)
    labelCounts={}
    for featVec in dataSet:
        # 每行数据的最后一个是类别
        currentLabel=featVec[-1]
        # 统计有多少个类以及每个类的数量
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    # 计算香农熵(Shannon Entropy)
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值
        shannonEnt-=prob*log(prob,2) # 累加每个类的熵值
    return shannonEnt

# 按照给定属性的给定取值提取数据集 
def splitDataSet(dataSet,axis,value):
    # 用来存储提取后的数据集
    retDataSet=[]
    # 如果样本的给定属性的取值恰好等于给定取值
    for featVec in dataSet:
        if featVec[axis]==value:
            # 保留此属性前的属性的取值
            reducedFeatVec =featVec[:axis]
            # 加上保留此属性后的属性的取值
            reducedFeatVec.extend(featVec[axis+1:])
            # 将保留前后属性取值的样本加到总的列表中
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集的划分方式
def chooseBestFeatureToSplit(dataSet):
    # 计算属性数量
    numFeatures = len(dataSet[0])-1
    # 计算数据集整体熵
    baseEntropy = calcShannonEnt(dataSet)
    # 用来存储最好的信息增益(bestInfoGain)和最好划分特征(bestFeature)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        # 建立该属性所有样本的取值的列表
        featList = [example[i] for example in dataSet]
        # 建立该属性所有可能的取值(即分类标签)的列表
        uniqueVals = set(featList)
        # 用来存储该属性的熵
        newEntropy = 0
        # 对于该属性的每一个分类标签
        for value in uniqueVals:
            # 得到该属性该分类标签的样本数据列表
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            # 按特征分类后的熵
            newEntropy +=prob*calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
        if (infoGain>bestInfoGain):   
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

#按分类后类别数量排序，消除噪音，比如：最后分类为2R1A，则判定为R；
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 创建树
def createTree(dataSet,labels):
    # 建立所有可能结果的列表
    classList=[example[-1] for example in dataSet]
    # 如果所有的结果都一致，则停止继续划分
    if classList.count(classList[0])==len(classList):
        return classList[0]
    # 如果遍历完所有属性仍有未分类的，则将其划分到次数最多的结果上
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    # 计算最好的数据集划分方式
    bestFeat=chooseBestFeatureToSplit(dataSet)
    # 得到最好的数据集划分方式的属性名称
    bestFeatLabel=labels[bestFeat]
    # 建立以该属性为根节点的树
    myTree={bestFeatLabel:{}}
    # 将这个属性从属性列表中删除（不再作为后续选最好划分方式的属性）
    del(labels[bestFeat])
    # 得到该属性所有样本的取值的列表
    featValues=[example[bestFeat] for example in dataSet]
    # 得到该属性所有可能的取值的列表
    uniqueVals=set(featValues)
    # 对于每一个可能的取值
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
                            (dataSet,bestFeat,value),subLabels)
    return myTree
 
 
if __name__=='__main__':
    fr = open('/Users/xinyuanhe/Desktop/working/2021美赛/模型/【正式】模型-机器学习-分类-决策树【hxy】/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'precript', 'astigmatic', 'tearRate']
    print(createTree(lenses, lensesLabels))
