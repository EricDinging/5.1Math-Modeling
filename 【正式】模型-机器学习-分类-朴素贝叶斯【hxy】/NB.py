# 导入numpy库
from numpy import *
# 导入正则表达式库
import re

# 创建一个包含在所有文档中出现的不重复次的列表
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

# 输入：vacabList是包含所有词的列表，inputSet是待建立文档向量的文档
# 输出：inputSet对应的文档向量
# 与bagOfWords2Vec不同的是，该向量记录的是每个单词在该文档中是否出现
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 输入：vacabList是包含所有词的列表，inputSet是待建立文档向量的文档
# 输出：inputSet对应的文档向量
# 与setOfWords2Vec不同的是，该向量记录的是每个单词在该文档中的出现次数
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 训练分类器
# 输入：trainMatrix是所有文档向量构成的文档矩阵，trainCategory是所有文档类别构成的文档类别向量
# 输出：p0Vect是分类为class0的各单词的概率矩阵，p1Vect是分类为class1的各单词的概率矩阵
# 输出：pAbusive是任意文档属于class0的概率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 为了避免其中一个概率值为0，最后乘积也为0
    # 将所有词的出现数初始化为1，并将分母初始化为2
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] # 向量相加
            p1Denom += sum(trainMatrix[i]) 
        else:
            p0Num += trainMatrix[i] # 向量相加
            p0Denom += sum(trainMatrix[i])
    # 为了解决*下溢出*（即很多很小的因子相乘被近似为0），对概率取对数
    p1Vect = log(p1Num/p1Denom)        
    p0Vect = log(p0Num/p0Denom)         
    return p0Vect,p1Vect,pAbusive

# 朴素贝叶斯分类
# 输入：vec2Classify待分类的向量
# 输出：分类类别
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 概率相乘（由于取了对数，变成相加）
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

# 输入：一封邮件内容
# 输出：小写的以除单词、数字外的任意字符串分割的长于2字符的单词列表
def textParse(bigString):
    import re
    #匹配非英文字母和数字
    regEx = re.compile('\\W+') 
    listOfTokens = regEx.split(bigString)
    # 小写，并过滤掉小于2的单词
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 测试主代码
def spamTest():
    docList=[]; classList = []; fullText =[]
    # 读入数据集
    for i in range(1,26):
        # 用‘gb18030’的编码方式打开txt文件，并忽视无法编码的特殊字符
        # wordList暂存当前文档的单词列表
        wordList = textParse(open('email/spam/%d.txt' % i,encoding='gb18030', errors='ignore').read())
        # 检查分割邮件的效果
        if i==1:
            print('Division result for email/spam/1.txt is:')
            print(wordList)
        # docList每个元素都是一个单词列表
        docList.append(wordList)
        # fullText每个元素都是一个单词
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,encoding='gb18030', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建一个包含在所有文档中出现的不重复次的单词列表
    vocabList = createVocabList(docList)
    # 创建测试集
    trainingSet = list(range(50)); testSet=[]
    # 随机选择10封邮件作为训练集，训练分类器，并从测试集中删去
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    # 将剩下测试集中的40封邮件分类
    for docIndex in testSet:        
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            # 打印错误分类的单词，可以根据这些找错误的邮件
            print("Classification error:")
            print(docList[docIndex])
    # 打印错误的比例
    print('The error rate is: ',float(errorCount)/len(testSet))

spamTest()
