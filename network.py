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
        for iteration in range(iterations):
            for x, y in zip(_x, _y):
                output = self.think(x)
                error = loss.mean_squared(output, y)
                print(output, error)
                self.backprop(error, x, learning_rate)

    def think(self, x):
        self.layers[0].forward(x)
        for n, layer in enumerate(self.layers[1:], start=1):
            prev_layer = self.layers[n - 1]
            layer.forward(prev_layer.output)
        return self.layers[-1].output

    def backprop(self, error, x, learning_rate):
        # Backprop output layer
        self.layers[-1].backward_out(error, self.layers[-2], learning_rate)

        # Backward propagate hidden layer (if big enough network)
        if len(self.layers) > 2:
            for n, layer in reversed(list(enumerate(self.layers[:-1]))):
                prev_layer = self.layers[n + 1]
                layer.backward(prev_layer, learning_rate)

        # Backprop input layer
        self.layers[0].backward_in(self.layers[1], x, learning_rate)


class Layer:
    def __init__(self, input_amount, neuron_amount):
        self.weights = np.random.randn(input_amount, neuron_amount) * 0.1
        self.prev_weights = self.weights
        self.biases = np.ones((1, neuron_amount))

    def forward(self, inputs):
        self.inputs = np.dot(inputs, self.weights.T) + self.biases
        self.output = activation.sigmoid(self.inputs)

    def backward(self, prev, learning_rate):
        self.gradients = activation.sigmoid_d(self.inputs) * (prev.gradients * prev.weights)

    def backward_in(self, prev, x, learning_rate):
        self.gradients = activation.sigmoid_d(self.inputs) * (prev.gradients * prev.weights)
        self.weights = self.weights + learning_rate * self.prev_weights + 0.25 * self.gradients * x
        self.prev_weights = self.weights

    def backward_out(self, error, next, learning_rate):
        self.gradients = activation.sigmoid_d(self.inputs) * error
        self.weights = self.weights + learning_rate * self.prev_weights + 0.25 * self.gradients * next.output
        self.prev_weights = self.weights


x = np.array([[0.1, 0.9]])
y = np.array([[0.9]])

nn = Network()

l1 = Layer(2, 2)
l1.weights = np.array([[-0.2, 0.1], [-0.1, 0.3]])
l1.prev_weights = np.array([[-0.2, 0.1], [-0.1, 0.3]])
l1.biases = np.array([[0.1, 0.1]])

l2 = Layer(2, 1)
l2.weights = np.array([[0.2, 0.3]])
l2.prev_weights = np.array([[0.2, 0.3]])
l2.biases = np.array([[0.2]])

nn.add(l1)
nn.add(l2)

nn.train(100000, 0.0001, x, y)
