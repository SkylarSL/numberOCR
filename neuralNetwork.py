# -*- coding: utf-8 -*-
import numpy as np
import random
#imports the input
import numInput

# network with layer size:
# input layer = 900,  30*30 pixels
# 1st hidden layer = 15
# output layer =  10

training = numInput.getMNIST();

class neuralNetwork:
    def __init__(self):
        self.weights = [np.random.randn(15, 728), np.random.randn(10, 15)]
        self.biases = [np.random.randn(15, 1), np.random.randn(10, 1)]

    def getOuput(self, a):
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
                self.update(miniBatch, trainingTime)
            if testData:
                print("Epoch" + i + ":" + self.evaluate(testData)
                      + "/" + nTest)
            else:
                print("Epoch" + format(i) + "complete")

    def update_wb(self, mini_batch, eta):
        gradient_b = [np.zeros(np.shape(b)) for b in self.weights]
        gradient_w = [np.zeros(np.shape(w)) for w in self.biases]

        for x, y in mini_batch:
            nudge_gb, nudge_gw = self.backprop(x, y)
            for i in range(len(gradient_b)):
                gradient_b[i] = gradient_b[i] + nudge_gb[i]
                gradient_w[i] = gradient_w[i] + nudge_gw[i]

        self.weights = self.weights - (eta / len(mini_batch)) * gradient_w
        self.biases = self.biases - (eta / len(mini_batch)) * gradient_b

    def backprop(self, actual, expect):
        bias = [np.zeros(b.shape) for b in self.biases]
        weight = [np.zeros(w.shape) for w in self.weights]

        activation = actual;
        activations = [actual];

        zvector = [];

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b;
            zvector.append(z);
            activation = sigmoid(z)
            activations.append(activation);

        delta_cost = self.cost(activations[2], expect) * primeSigmoid(zvector[2])
        bias[-1] = delta_cost;
        weight[-1] = np.dot(delta_cost, activations[1].transpose())

        delta_cost = np.dot(self.weights[1].transpose(), delta_cost) * \
                     primeSigmoid(zvector[-1])
        bias[-1] = delta_cost
        weight[-1] = np.dot(delta_cost, activations[0].transpose())

        return (bias, weight);

    def cost(self, actual_value, expected_value):
        return (actual_value - expected_value);

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def primeSigmoid(x):
    return sigmoid(x)*(1.0-sigmoid(x))

samp = neuralNetwork()
out = samp.getOuput(np.matrix(np.arange(728).reshape(728, 1)))
print("output layer: ", "\n", out)
print("The number is likely: ", np.argmax(out, 0)[0, 0])
