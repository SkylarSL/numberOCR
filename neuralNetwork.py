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
test = numInput.resize();

class neuralNetwork:
    def __init__(self):
        self.weights = [np.random.randn(30, 784), np.random.randn(10, 30)]
        self.biases = [np.random.randn(30, 1), np.random.randn(10, 1)]

    def getOutput(self, a):
        for i in range(len(self.weights)):
            a = sigmoid(np.dot(self.weights[i], a) + self.biases[i])
        return a

    def SGD(self, trainingData, sets, miniBatchSize, trainingTime, testData = None):
        if testData:
            nTest = len(testData)
        n = len(trainingData)
        for i in range(sets):
            random.shuffle(trainingData);
            miniBatches = [trainingData[k: k + miniBatchSize]
                           for k in range(0, n, miniBatchSize)]
            for miniBatch in miniBatches:
                self.update_wb(miniBatch, trainingTime)
            if testData:
                print("set" + str(i) + ":" + str(self.evaluate(testData))
                      + "/" + str(nTest))
            else:
                print("sets" + format(i) + "complete")

    def update_wb(self, mini_batch, eta):
        gradient_b = [np.zeros(np.shape(b)) for b in self.biases]
        gradient_w = [np.zeros(np.shape(w)) for w in self.weights]
        for x, y in mini_batch:
            nudge_gb, nudge_gw = self.backprop(x, y)
            for i in range(len(gradient_b)):
                gradient_b[i] = gradient_b[i] + nudge_gb[i]
                gradient_w[i] = gradient_w[i] + nudge_gw[i]
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (eta / len(mini_batch)) * gradient_w[i]
            self.biases[i] = self.biases[i] - (eta / len(mini_batch)) * gradient_b[i]

    def backprop(self, actual, expect):
        grad_bias = [np.zeros(np.shape(b)) for b in self.biases]
        grad_weight = [np.zeros(np.shape(w)) for w in self.weights]

        activations = [actual];
        non_sigmoid = [];

        for i in range(len(self.weights)):
            node_value = np.dot(self.weights[i],actual)+self.biases[i]
            non_sigmoid.append(node_value)
            actual = sigmoid(node_value)
            activations.append(actual)

        delta_cost = np.multiply(self.cost(activations[2], expect), primeSigmoid(non_sigmoid[1]))
        grad_bias[1] = delta_cost;
        grad_weight[1] = delta_cost * activations[1].transpose()

        delta_cost = np.multiply(self.weights[1].transpose()*delta_cost, primeSigmoid(non_sigmoid[0]))
        grad_bias[0] = delta_cost
        grad_weight[0] = delta_cost * activations[0].transpose()

        return (grad_bias, grad_weight);

    def evaluate(self, test_data):
        correct = 0
        for i in test_data:
            if np.argmax(self.getOutput(i[0]), 0) == np.argmax(i[1], 0):
                correct += 1
        return correct

    def test(self, input):
        print(len(input))
        print(len(input[0]))
        print(np.argmax(self.getOutput(input[0]), 0))

    def cost(self, actual_value, expected_value):
        return (actual_value - expected_value);

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def primeSigmoid(x):
    return np.multiply(sigmoid(x),(1.0-sigmoid(x)))

samp = neuralNetwork()
samp.SGD(training[:50000], 10, 10, 0.6, training[50000:])
samp.test(test)

#writes to the text file, saves weights and biases
'''
weight_file = open("weights.txt", "w")
bias_file = open("biases.txt", "w")
weights = samp.weights
biases = samp.biases

for i in range(len(weights)):
    weight_file.write("\n \nweight layer " + str(i+1) + " : ")
    for j in range(len(weights[i])):
        weight_file.write(str(weights[i][j]))

for i in range(len(biases)):
    bias_file.write("\n \nbias layer " + str(i+1) + " : ")
    for j in range(len(biases[i])):
        bias_file.write(str(biases[i][j]))

weight_file.close()
bias_file.close()
'''
