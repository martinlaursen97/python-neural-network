import numpy as np
import activation
import loss

np.random.seed(0)


class Network:
    def __init__(self, layers):
        self.weights = [np.random.rand(layers[i - 1], layers[i]) for i in range(1, len(layers))]
        self.prev_weights = self.weights

        self.inputs = []
        self.outputs = []
        self.gradients = []

    def forward(self, X):
        inputs = []
        outputs = []

        prev_input = X
        for w in self.weights:
            v = np.dot(prev_input, w) + 0.1
            y = activation.sigmoid(v)
            prev_input = y

            inputs.append(v)
            outputs.append(y)
        self.inputs = inputs
        self.outputs = outputs

        return self.outputs[-1]

    def backward(self, error, learning_rate):
        gradients = [0] * len(self.weights)

        output_gradient = activation.sigmoid_d(self.inputs[-1]) * error
        gradients[-1] = output_gradient

        for i in range(len(self.weights) - 2, -1, -1):
            gradients[i] = activation.sigmoid_d(self.inputs[i]) * (gradients[i + 1] * self.weights[i + 1])

        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * self.prev_weights[i] + 0.2 * gradients[i] * self.inputs[i]

        self.prev_weights = self.weights

        self.gradients = gradients

    def train(self, iterations, learning_rate, inputs, targets):
        for i in range(iterations):

            for X, t in zip(inputs, targets):
                output = self.forward(X)

                error = loss.mean_squared(output, t)


                self.backward(error, learning_rate)

                print(t, output, error)


X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
t = np.array([[0], [1], [1], [1]])

nn = Network([2, 2, 1])
nn.train(20, 0.01, X, t)
