import numpy as np
import activation
import loss

np.random.seed(0)


class Network:
    def __init__(self, layers):
        self.weights = [np.random.rand(layers[i - 1], layers[i]) for i in range(1, len(layers))]
        self.inputs = []
        self.outputs = []

    def forward(self, X):
        prev = X
        for w in self.weights:
            z = np.dot(prev, w)
            a = activation.sigmoid(z)
            prev = a

            self.inputs.append(z)
            self.outputs.append(a)

        return self.outputs[-1]

    def train(self, iterations, inputs, targets):
        for i in range(iterations):
            for X, t in zip(inputs, targets):
                output = self.forward(X)

                error = loss.mean_squared(output, t)

                print(error)


X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
t = np.array([[0], [1], [1], [1]])
nn = Network([2, 2, 1])
nn.train(1, X, t)
