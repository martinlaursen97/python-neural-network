import numpy as np
import activation as a

np.random.seed(0)


class Network:
    def __init__(self, layers):
        self.layers = layers

    def insert_training_inputs(self, training_inputs):
        self.training_inputs = training_inputs

    def insert_training_targets(self, training_targets):
        self.training_targets = training_targets

    def train(self):
        pass


class Layer:
    def __init__(self, input_amount, layer_size):
        # Every neuron will have 1 weight for every neuron in the previous layer
        self.weights = np.random.randn(input_amount, layer_size)

        # 1 bias for every neuron
        self.biases = np.zeros((1, layer_size))

    def forward(self, inputs):
        # Need to calculate the outputs (w*i +b) of every neuron when forward propagating
        self.output = a.sigmoid(np.dot(inputs, self.weights) + self.biases)


l_in = Layer(2, 2)
l_out = Layer(2, 1)

input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
input_targ = np.array([[0], [1], [1], [1]])

l_in.forward(input_data)
l_out.forward(l_in.output)

network = Network([l_in, l_out])
network.insert_training_inputs(input_data)
network.insert_training_targets(input_targ)