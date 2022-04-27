import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

# def createDataSet():
#     group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    num = len(arrayOLines)
    dataset = np.zeros((num, 3))
    y = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listfromline = line.split('\t')
        dataset[index, :]  = listfromline[0:3]  # the dimension of the matrix is 3
        y.append(listfromline[-1])
        index +=1
    return dataset, y

def autoNorm(dataset):
    num = dataset.shape[1]
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    normdataset = (dataset - np.reshape(minVals, (1, num)))/ np.reshape(maxVals - minVals, (1, num))
    return normdataset

def test(Xtest, ytest, Xtrain, ytrain, k):
    ntest = Xtest.shape[0]
    count = 0
    for i in range(ntest):
        if classify0(Xtest[i], Xtrain, ytrain, k) != ytest[i]:
            count += 1
    error = count / ntest * 100
    print('Error: ', error)



def classify0(x, dataset, labels, k):
    distance = np.sqrt(np.sum((dataset - x) ** 2, axis=1))
    sortdistind = distance.argsort()

    classcount = {}
    for i in range(k):
        label = labels[sortdistind[i]]
        classcount[label] = classcount.get(label, 0) + 1
    sortedclasscount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]

labeldict = {
    "didntLike":1,
    "smallDoses":2,
    "largeDoses":3,
}
X, y = file2matrix('datingTestSet.txt')
for i in range(len(y)):
    y[i] = labeldict[y[i]]
X = autoNorm(X)
n = X.shape[0]
ntrain = int(n * 0.9)
Xtrain = X[:ntrain, :]
ytrain = y[:ntrain]
Xtest = X[ntrain:, :]
ytest = y[ntrain:]

# plt.scatter(X[:,0], X[:,1], s=15 * np.array(y), c=15 * np.array(y))
# plt.show()

test(Xtest, ytest, Xtrain, ytrain, 3)

x = [0, 0, 0]
ypredict = classify0(x, Xtrain, ytrain, 3)
print(ypredict)