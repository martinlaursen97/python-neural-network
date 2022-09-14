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
                self.backprop(error)

    def think(self, input):
        self.layers[0].forward(input)
        for n, layer in enumerate(self.layers[1:], start=1):
            prev_layer = self.layers[n - 1]
            layer.forward(prev_layer.output)
        return self.layers[-1].output

    def backprop(self, error):
        self.layers[-1].backward()

        # Backward propagate remaining layers
        for n, layer in reversed(list(enumerate(self.layers[:-1]))):
            prev_layer = self.layers[n + 1]
            layer.backward()


class Layer:
    def __init__(self, input_amount, neuron_amount):
        self.weights = np.random.randn(input_amount, neuron_amount) * 0.1
        self.biases = np.ones((1, neuron_amount))

    def forward(self, inputs):
        self.inputs = np.dot(inputs, self.weights) + self.biases
        self.output = activation.sigmoid(self.inputs)

    def backward(self):
        pass


x = np.array([[0.1, 0.9]])
y = np.array([[0.9]])

nn = Network()

l1 = Layer(2, 2)
l1.weights = np.array([[-0.2, 0.1], [0.1, 0.3]])
l1.biases = np.array([[0.1, 0.1]])

l2 = Layer(2, 1)
l2.weights = np.array([0.2, 0.3])
l2.biases = np.array([[0.2]])

nn.add(l1)
nn.add(l2)

nn.train(10, 0.01, x, y)
