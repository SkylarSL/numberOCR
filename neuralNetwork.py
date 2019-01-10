# -*- coding: utf-8 -*-
import numpy as np
import random
#imports the input
import NumInput

# network with layer size:
# input layer = 784
# 1st hidden layer = 30
# output layer =  10

training = NumInput.getMNIST();

class neuralNetwork:
    def __init__(self):
        self.weights = [np.random.randn(30,784), np.random.randn(10,30)]
        self.biases = [np.random.randn(30, 1), np.random.randn(10, 1)]

    def getOutput(self, a):
        for i in range(len(self.weights)):
            a = sigmoid(np.dot(self.weights[i], a) + self.biases[i])
        return a

    def SGD(self, trainingData, totalData, miniBatchSize, trainingTime, testData = None):
        if testData:
            nTest = len(testData)
        n = len(trainingData)
        for i in range(totalData):
            random.shuffle(trainingData);
            miniBatches = [trainingData[k: k + miniBatchSize]
                           for k in range(0, n, miniBatchSize)]
            for miniBatch in miniBatches:
                self.update_wb(miniBatch, trainingTime)
            if testData:
                print("Epoch" + str(i) + ":" + str(self.evaluate(testData))
                      + "/" + str(nTest))
            else:
                print("Epoch" + format(i) + "complete")
    def update_wb(self,mini_batch,eta):
        gradient_b = [np.zeros(np.shape(b)) for b in self.biases]
        gradient_w = [np.zeros(np.shape(w)) for w in self.weights]
        for x,y in mini_batch:
            nudge_gb, nudge_gw = self.backprop(x,y)
            for i in range(len(gradient_b)):
                #print(np.shape(gradient_w[i]),np.shape(nudge_gw[i]))
                gradient_b[i] = gradient_b[i] + nudge_gb[i]
                gradient_w[i] = gradient_w[i] + nudge_gw[i]
        for i in range(len(self.weights)):
            #print(len(self.weights[i][1]),len(gradient_w[i]))
            self.weights[i] = self.weights[i] - ((eta/len(mini_batch))*gradient_w[i])
            self.biases[i] = self.biases[i] - ((eta/len(mini_batch))*gradient_b[i])


    def backprop(self, a, y):
        gradient_b = [np.zeros(np.shape(b)) for b in self.biases]
        gradient_w = [np.zeros(np.shape(w)) for w in self.weights]

        activation = [a]
        non_sigmoid = []
        for i in range(len(self.weights)):
            temp = np.dot(self.weights[i],a)+self.biases[i]
            #print("temp",temp)
            non_sigmoid.append(temp)
            a=sigmoid(temp)
            activation.append(a)
        

        delta = np.multiply((activation[2]-y),primeSigmoid(non_sigmoid[1]))
        gradient_w[1] = delta*(activation[1]).transpose()
        gradient_b[1] = delta
        #print("l",np.shape(gradient_w[1]))
        delta = np.multiply(self.weights[1].transpose()*delta,primeSigmoid(non_sigmoid[0]))
        gradient_b[0] = delta
        gradient_w[0] = delta*activation[0].transpose()
        
        return(gradient_b,gradient_w)
    

    def evaluate(self, test_data):
        correct = 0
        for i in test_data:
            if np.argmax(self.getOutput(i[0]),0)[0]==np.argmax(i[1],0)[0]:
                correct+=1
        return correct

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def primeSigmoid(x):
    return np.multiply(sigmoid(x),(1.0-sigmoid(x)))

samp = neuralNetwork()
samp.SGD(training[:50000],3,10,3.0,training[50000:])
#outfile = open("weightandbias.txt",'w')
#outfile.write(str(np.array(samp.weights)))
#outfile.write('\n')
#outfile.write(str(samp.biases))
#outfile.close()

