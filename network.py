import numpy as np
import activation
import loss

np.random.seed(0)


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def train(self, iterations, learning_rate, _x, _y):
        for i in range(iterations):
            for x, y in zip(_x, _y):
                output = self.predict(x)
                error = loss.difference(output, y)

                if i % 10 == 0:
                    print(i, '-', x, y, output, loss.mean_squared(output, y))

                self.backprop(error, x, learning_rate)

    def predict(self, x):
        self.layers[0].forward(x)
        for n, layer in enumerate(self.layers[1:], start=1):
            prev_layer = self.layers[n - 1]
            layer.forward(prev_layer.output)
        return self.layers[-1].output

    def backprop(self, error, x, learning_rate):
        self.layers[-1].backward_out(error, self.layers[-2], learning_rate)

        if len(self.layers) > 2:
            for n in range(len(self.layers) - 2, 0, -1):
                layer = self.layers[n]
                prev_layer = self.layers[n + 1]
                layer.backward_h(prev_layer, learning_rate)

        self.layers[0].backward_in(self.layers[1], x, learning_rate)


class Layer:
    def __init__(self, input_amount, neuron_amount):
        self.weights = np.random.randn(input_amount, neuron_amount) * 0.1
        self.prev_weights = self.weights
        self.biases = np.zeros((1, neuron_amount))
        self.prev_biases = self.biases
        self.thresh_hold = 0.25

    def forward(self, inputs):
        self.inputs = np.dot(inputs, self.weights) + self.biases
        self.output = activation.sigmoid(self.inputs)

    def backward_h(self, prev, learning_rate):
        self.set_gradients(np.dot(prev.gradients, prev.weights.T), self.inputs)
        self.adjust_weights(self.output, learning_rate)
        self.adjust_biases(learning_rate)

    def backward_in(self, prev, x, learning_rate):
        self.set_gradients(np.dot(prev.gradients, prev.weights.T), self.inputs)
        self.adjust_weights(np.array([x]).T, learning_rate)
        self.adjust_biases(learning_rate)

    def backward_out(self, error, next, learning_rate):
        self.set_gradients(error, self.inputs)
        self.adjust_weights(next.output.T, learning_rate)
        self.adjust_biases(learning_rate)

    def set_gradients(self, error, x):
        self.gradients = error * activation.sigmoid_d(x)

    def adjust_weights(self, x, learning_rate):
        self.weights = self.weights + learning_rate * self.prev_weights + self.thresh_hold * self.gradients * x
        self.prev_weights = self.weights

    def adjust_biases(self, learning_rate):
        self.biases = self.biases + learning_rate * self.prev_biases + self.thresh_hold * self.gradients
        self.prev_biases = self.biases


x = [[0, 0], [1, 0], [0, 1], [1, 1]]
y = [[0], [1], [1], [0]]

nn = Network()

l1 = Layer(2, 10)
l2 = Layer(10, 10)
l3 = Layer(10, 1)

nn.add(l1)
nn.add(l2)
nn.add(l3)

nn.train(1000, 0.001, x, y)

output = nn.predict([[0, 0]])
print(output)
