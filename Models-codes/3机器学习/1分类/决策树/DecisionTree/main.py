from math import log
import operator
import treePlotter

def calcShannonEnt(dataset):
    n = len(dataset)
    labelcounts = {}
    for vec in dataset:
        label = vec[-1]
        labelcounts[label] = labelcounts.get(label, 0) + 1
    entropy = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key]) / n
        entropy -= prob * log(prob, 2)
    return entropy

def createDataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [1, 1, 'yes'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    featurelabel = ['no surfacing', 'flippers']
    return dataset, featurelabel

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = [*featVec[:axis], *featVec[axis+1:]]
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    n = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(n):
        featList = [example[i] for example in dataSet]
        vals = set(featList)
        newEntropy = 0.0
        for value in vals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, featLabel):
    y = [example[-1] for example in dataSet]
    if y.count(y[0]) == len(y):
        return y[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(y)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = featLabel[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(featLabel[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = featLabel[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


dat, features = createDataset()
myTree = createTree(dat, features)
print(treePlotter.getNumLeafs(myTree))