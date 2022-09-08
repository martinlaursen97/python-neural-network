import numpy as np

np.random.seed(0)


class Layer:
    def __init__(self, inputs, neurons):
        # Every neuron will have 1 weight for every neuron in the previous layer
        self.weights = np.random.randn(inputs, neurons)

        # 1 bias for every neuron
        self.biases = np.zeros((1, neurons))
