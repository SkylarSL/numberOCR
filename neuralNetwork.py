# -*- coding: utf-8 -*-
import numpy as np

#imports the input
import numInput

# network with layer size:
# input layer = 900,  30*30 pixels
# 1st hidden layer = 15
# output layer =  10

class neuralNetwork:
    def __init__(self):
        self.weights = [np.random.randn(15, 900), np.random.randn(10, 15)]
        self.biases = [np.random.randn(15, 1), np.random.randn(10, 1)]

    def getOuput(self, a):
        for i in range(len(self.weights)):
            a = sigmoid(np.dot(self.weights[i], a) + self.biases[i])
        return a

    def backprop(self, actual, expect):
        return "";

    def cost(self, actual_value, expected_value):
        return (actual_value - expected_value);

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def primeSigmoid(x):
    return sigmoid(x)*(1.0-sigmoid(x))


samp = neuralNetwork()
out = samp.getOuput(np.matrix(np.arange(900).reshape(900, 1)))
print("output layer: ", "\n", out)
print("The number is likely: ", np.argmax(out, 0)[0, 0] + 1)
