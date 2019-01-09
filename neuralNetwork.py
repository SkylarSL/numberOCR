# -*- coding: utf-8 -*-
import numpy as np
#imports the input
import numInput

# network with layer size:
# input layer = 900,  30*30 pixels
# 1st hidden layer = 15
# output layer =  10

images = numInput.getMNIST()[0];
value = numInput.getMNIST()[1];

class neuralNetwork:
    def __init__(self):
        self.weights = [np.random.randn(15, 728), np.random.randn(10, 15)]
        self.biases = [np.random.randn(15, 1), np.random.randn(10, 1)]

    def getOuput(self, a):
        for i in range(len(self.weights)):
            a = sigmoid(self.weights[i]*a + self.biases[i])
        return a

    def backprop(self, a, b):
        gradient_b = [np.zeros(np.shape(b)) for b in self.weights]
        gradient_w = [np.zeros(np.shape(w)) for w in self.biases]
        
        activation = [a]
        non_sigmoid = []
        for i in range(len(self.weights)):
            temp = np.dot(self.weights[i],a)+b
            non_sigmoid.append(temp)
            activation.append(sigmoid(temp))
        
        delta = (activation[1]-y)
        gradient_w[1] = delta*primeSigmoid(non_sigmoid[1]).transpose()
        gradient_b[1] = delta
        
        delta = self.weights[1].transpose*delta*primeSigmoid(non_sigmoid[0])
        gradient_w[0] = delta
        gradient_b[0] = delta*activation[0].transpose
        
        return (gradient_w,gradient_b)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def primeSigmoid(x):
    return sigmoid(x)*(1.0-sigmoid(x))

samp = neuralNetwork()
out = samp.getOuput(np.matrix(np.arange(728).reshape(728, 1)))
print("output layer: ", "\n", out)
print("The number is likely: ", np.argmax(out, 0)[0, 0] + 1)
