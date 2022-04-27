import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(x, dataset, labels, k):
    distance = np.sqrt(np.sum((dataset - x) ** 2, axis=1))
    sortdistind = distance.argsort()

    classcount = {}
    for i in range(k):
        label = labels[sortdistind[i]]
        classcount[label] = classcount.get(label, 0) + 1
    sortedclasscount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]

X, y = createDataSet()
x = [0, 0]
ypredict = classify0(x, X, y, 3)
print(ypredict)