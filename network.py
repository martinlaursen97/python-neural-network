import numpy as np
import pandas as pd
import activation
import loss
from frame import Frame


# np.random.seed(1)


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def train(self, iterations, learning_rate, _x, _y, mod=100, verbose=True, loss_f='difference'):
        for i in range(iterations):
            for x, y in zip(_x, _y):
                output = self.predict(x)

                error = 0
                if loss_f == 'difference':
                    error = loss.difference(output, y)
                elif loss_f == 'categorical_crossentropy':
                    error = loss.categorical_crossentropy(output, y)

                if verbose:
                    if i % mod == 0:
                        print(error)

                self.backprop(error, x, learning_rate)

    def predict(self, x):
        self.layers[0].forward(x)
        for n, layer in enumerate(self.layers[1:], start=1):
            prev_layer = self.layers[n - 1]
            layer.forward(prev_layer.output)
        return self.layers[-1].output

    def backprop(self, error, x, learning_rate):
        self.layers[-1].backward(error, self.layers[-2].output.T, learning_rate)

        if len(self.layers) > 2:
            for n in range(len(self.layers) - 2, 0, -1):
                layer = self.layers[n]
                prev_layer = self.layers[n + 1]
                layer_error = np.dot(prev_layer.gradients, prev_layer.weights.T)
                layer.backward(layer_error, layer.output, learning_rate)

        inp_layer_error = np.dot(self.layers[1].gradients, self.layers[1].weights.T)
        self.layers[0].backward(inp_layer_error, np.array([x]).T, learning_rate)


class Layer:
    def __init__(self, input_amount, neuron_amount, activation_f='sigmoid'):
        self.weights = np.random.randn(input_amount, neuron_amount) * 0.1
        self.prev_weights = self.weights
        self.biases = np.zeros((1, neuron_amount))
        self.prev_biases = self.biases
        self.thresh_hold = 0.25
        self.activation_f = activation_f

    def forward(self, inputs):
        self.inputs = np.dot(inputs, self.weights) + self.biases
        if self.activation_f == 'sigmoid':
            self.output = activation.sigmoid(self.inputs)
        elif self.activation_f == 'softmax':
            self.output = loss.softmax(self.inputs)

    def backward(self, error, inp, learning_rate):
        self.set_gradients(error, self.inputs)
        self.adjust_weights(inp, learning_rate)
        self.adjust_biases(learning_rate)

    def set_gradients(self, error, x):
        self.gradients = error * activation.sigmoid_d(x)

    def adjust_weights(self, x, learning_rate):
        self.weights = self.weights + learning_rate * self.prev_weights + self.thresh_hold * self.gradients * x
        self.prev_weights = self.weights

    def adjust_biases(self, learning_rate):
        self.biases = self.biases + learning_rate * self.prev_biases + self.thresh_hold * self.gradients
        self.prev_biases = self.biases


frame = Frame('mnist_test.csv')
frame.transform(0, 1)
frame.normalize()

x, y = frame.get_x(), pd.get_dummies(frame.get_y()).values

nn = Network()

nn.add(Layer(784, 784, activation_f='sigmoid'))
nn.add(Layer(784, 15, activation_f='sigmoid'))
nn.add(Layer(15, 15, activation_f='sigmoid'))
nn.add(Layer(15, 10, activation_f='softmax'))
nn.train(1, 0.001, x, y, verbose=True, loss_f='categorical_crossentropy')

pred = nn.predict(x[2])
print(np.argmax(pred), frame.get_y()[2], pred)
