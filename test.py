def backward(self, prev, next, learning_rate):
    self.gradients = activation.sigmoid_d(self.inputs) * (prev.gradients * prev.weights)
    self.weights = self.weights + learning_rate * self.prev_weights + 0.25 * self.weights * next.output.T
    self.prev_weights = self.weights

    self.biases = self.biases + learning_rate * self.prev_biases + 0.25 * self.gradients
    self.prev_biases = self.biases


def backward_in(self, prev, x, learning_rate):
    self.gradients = activation.sigmoid_d(self.inputs) * (prev.gradients * prev.weights.T)
    # print((self.weights + learning_rate * self.prev_weights + 0.25).shape)
    # print(self.gradients.shape)
    # print(np.array(x).shape)
    self.weights = self.weights + learning_rate * self.prev_weights + 0.25 * self.gradients * x
    self.prev_weights = self.weights

    self.biases = self.biases + learning_rate * self.prev_biases + 0.25 * self.gradients
    self.prev_biases = self.biases


def backward_out(self, error, next, learning_rate):
    self.gradients = activation.sigmoid_d(self.inputs) * error
    self.weights = self.weights + learning_rate * self.prev_weights + 0.25 * self.gradients * next.output.T
    self.prev_weights = self.weights

    self.biases = self.biases + learning_rate * self.prev_biases + 0.25 * self.gradients
    self.prev_biases = self.biases