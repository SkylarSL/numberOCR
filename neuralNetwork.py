# -*- coding: utf-8 -*-
import numpy as np


#network with layer size:
#input layer = 900,  30*30 pixels
#1st hidden layer = 15
#output layer =  10


class neuralNetwork:
    def __init__(self):
        self.weights = [np.random.randn(15,900),np.random.randn(10,15)]
        self.biases = [np.random.randn(15,1),np.random.randn(10,1)]
    
    
    def getOuput(self,a):
        for i in range(len(self.weights)):
            a = self.sigmoid(np.dot(self.weights[i],a)+self.biases[i])
        return a

    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    
    
samp = neuralNetwork()
print(samp.getOuput(np.matrix(np.arange(900).reshape(900,1))))