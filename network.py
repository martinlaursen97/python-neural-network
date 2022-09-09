import numpy as np
import activation as a
import loss as l


# np.random.seed(0)


class Network:
    def __init__(self, layers):
        self.layers = layers

    def insert_training_inputs(self, training_inputs):
        self.training_inputs = training_inputs

    def insert_training_targets(self, training_targets):
        self.training_targets = training_targets

    def train(self, iterations):
        for iteration in range(iterations):
            for i, t in zip(self.training_inputs, self.training_targets):

                # Insert training data into input layer
                self.layers[0].forward(i)

                # Forward propagate remaining layers
                for n, layer in enumerate(self.layers[1:], start=1):
                    prev_layer = self.layers[n - 1]
                    layer.forward(prev_layer.output)

                # Output of the output layer ([-1] == last layer)
                actual_output = self.layers[-1].output

                # Calculate loss
                loss = l.mean_squared(actual_output, t)

                print(i, t, actual_output, loss)

                # Calculate gradient of output layer
                self.layers[-1].calc_gradients(loss)

                # Backward propagate remaining layers
                for n, layer in reversed(list(enumerate(self.layers[:-1]))):
                    prev_layer = self.layers[n + 1]
                    layer.calc_gradients(prev_layer.gradients)

                    #print(layer.gradients)


class Layer:
    def __init__(self, input_amount, layer_size):
        # Every neuron will have 1 weight for every neuron in the previous layer
        self.weights = np.random.randn(input_amount, layer_size)

        # 1 bias for every neuron
        self.biases = np.zeros((1, layer_size))

    def forward(self, inputs):
        # Need to calculate the outputs (w*i +b) of every neuron when forward propagating
        output = np.dot(inputs, self.weights) + self.biases
        self.inputs = output
        self.output = a.sigmoid(output)

    def calc_gradients(self, prev):
        # Prev layer gradients = loss if its the output layer

        self.gradients = a.sigmoid(self.inputs) * (1 - a.sigmoid(self.inputs)) * prev


l_in = Layer(2, 2)
l_h = Layer(2, 2)
l_out = Layer(2, 1)

input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
input_targ = np.array([[0], [1], [1], [1]])

network = Network([l_in, l_h, l_out])
network.insert_training_inputs(input_data)
network.insert_training_targets(input_targ)
network.train(200)
