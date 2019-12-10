import numpy as np
import math

embedDimension = 300
contextSize = 6
learningRate = 0.01
negativeSampleSize = 6
epochs = 20

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def neuralNets(vocabDimension, embedDimension):
    layer1 = np.random.uniform(low=-0.5 / embedDimension, high=0.5 / embedDimension,
                               size=(vocabDimension, embedDimension))
    layer2 = np.zeros(shape=(vocabDimension, embedDimension))
    return layer1, layer2

def saveFile(vocabWords, layer, fileName):
    f = open(fileName, 'w')
    for i in range(len(vocabWords)):
        s = str(vocabWords[i])
        for j in range(300):
            s = s + " " + str(layer[i][j])
        s = s + "\n"
        f.write(s)

class Vocabulary:
    def __init__(self, vocabFile):
        self.vocabFile = vocabFile
        self.vocabWords = []
        self.vocabToIndex = {}
        self.corpusWordCount = {}
        self.vocabWordsFunc()

    def vocabWordsFunc(self):
        f = open(self.vocabFile, 'r')
        for l in f:
            word = l.strip('\n')
            self.vocabToIndex[word] = len(self.vocabWords)
            self.vocabWords.append(word)
            self.corpusWordCount[word] = 0
        self.vocabWordsSet = set(x for x in self.vocabWords)

    def corpusWordsFunc(self, corpusFile):
        f = open(corpusFile, 'r')
        self.corpusWords = []
        for l in f:
            self.corpusWords.extend(l.split())

    def freqCorpusWords(self):
        for index in range(len(self.corpusWords)):
            word = self.corpusWords[index]
            if word in self.vocabWordsSet:
                self.corpusWordCount[word] = self.corpusWordCount[word] + 1
            else:
                self.corpusWords[index] = 'UNKNOWN'

    def buildNegativeTable(self):
        power = 0.75
        norm = np.sum(np.power(self.corpusWordCount[word], power) for word in self.corpusWordCount if word != 'UNKNOWN')
        self.negTableSize = np.power(10, 8)
        self.negTable = np.zeros(self.negTableSize, dtype=np.int32)
        i = 0
        prob = 0
        for word, count in self.corpusWordCount.items():
            if (word != 'UNKNOWN'):
                prob = prob + (np.power(count, power) / norm)
                wordIndex = self.vocabToIndex[word]
                while i < self.negTableSize and float(i) / self.negTableSize < prob:
                    self.negTable[i] = wordIndex
                    i += 1

    def sampleNegative(self, count, seed):
        np.random.seed(seed)
        indices = np.random.randint(low=0, high=self.negTableSize, size=count)
        return [self.vocabWords[self.negTable[i]] for i in indices]




class network:
    def __init__(self, vocab, embedDimension, contextSize, learningRate, negativeSampleSize, epochs):

        self.embedDimension = embedDimension
        self.contextSize = contextSize
        self.negativeSampleSize = negativeSampleSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.vocab = vocab
        self.vocabDimension = len(self.vocab.vocabWords)
        self.layer1, self.layer2 = neuralNets(self.vocabDimension, self.embedDimension)

    def buildTable(self):
        alpha = self.learningRate
        currNegSampleSize = self.contextSize * self.negativeSampleSize
        for i in range(epochs):
            currCost = 0
            for index in range(self.contextSize, len(self.vocab.corpusWords) - self.contextSize):
                centerWord = self.vocab.corpusWords[index]
                if centerWord != 'UNKNOWN':
                    centerIndex = self.vocab.vocabToIndex[centerWord]
                    contextStart = index - self.contextSize
                    contextEnd = index + self.contextSize + 1
                    context = self.vocab.corpusWords[contextStart:index] + self.vocab.corpusWords[
                                                                           contextStart + 1:contextEnd]
                    negSamples = self.vocab.sampleNegative(currNegSampleSize, index)
                    layer1S = self.layer1[centerIndex]
                    summation = np.zeros(embedDimension)
                    currError = 1
                    positiveClassifiers = [(contextWord, 1) for contextWord in context]
                    negativeClassifiers = [(negWord, 0) for negWord in negSamples]

                    for classifierWord, value in positiveClassifiers:
                        if classifierWord != 'UNKNOWN':
                            classifierIndex = self.vocab.vocabToIndex[classifierWord]
                            layer2C = self.layer2[classifierIndex]
                            z = np.dot(layer1S, layer2C)
                            observed = sigmoid(z)
                            currError = currError / observed
                            EI = alpha * (observed - value)
                            summation += EI * layer2C
                            self.layer2[classifierIndex] = layer2C - EI * layer1S

                    for classifierWord, value in negativeClassifiers:
                        if classifierWord != 'UNKNOWN':
                            classifierIndex = self.vocab.vocabToIndex[classifierWord]
                            layer2C = self.layer2[classifierIndex]
                            z = np.dot(layer1S, layer2C)
                            observed = sigmoid(z)
                            currError = currError * observed
                            EI = alpha * (observed - value)
                            summation += EI * layer2C
                            self.layer2[classifierIndex] = layer2C - EI * layer1S

                    currCost += math.log(currError + 1e-9)
                    self.layer1[centerIndex] = layer1S - summation
            alpha = alpha / 2