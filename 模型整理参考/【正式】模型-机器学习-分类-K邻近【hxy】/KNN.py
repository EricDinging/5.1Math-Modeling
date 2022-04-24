from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

# 读取数据
def file2matrix(filename):
    # 打开文件并读取每一行
    fr = open(filename)
    arrayOfLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOfLines)
    # 创建返回的Numpy矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    # 返回数据和标签
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

# 标准化数据
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals

# KNN算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 计算待求的跟数据集的差
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    # 按递增次序排序
    sortedDistIndices = distances.argsort()
    # 统计前k个里面每个label的数量
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    # 将label数量从大到小排序（用itemgetter方法，按照第二个元素数量的次序对元组进行排序）
    sortedClassCount = sorted(classCount,
                              key = operator.itemgetter(1), reverse = True)
    # 选出最多次的label
    return sortedClassCount[0]

# 约会数据KNN分类主函数
def datingClassTest():
    hoRatio = 0.50
    # 读取数据
    datingDataMat, datingLabels = file2matrix('/Users/xinyuanhe/Desktop/working/2021美赛/模型/模型-机器学习-分类-K邻近【hxy】/datingTestSet.txt')
    # 标准化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    # KNN算法
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],3)
        print("The classifier came back with: %s, the real answer is: %s" %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    # 计算错误率，检验KNN分类效果
    print("The total error rate is: %f" %(errorCount/float(numTestVecs)))

datingClassTest()
