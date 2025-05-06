import numpy as np


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) / np.sqrt(input_size)
        self.biases = np.zeros((output_size, 1))

    def forward(self, input_data):
        self.input_data = input_data
        return np.dot(self.weights, input_data) + self.biases

    def backward(self, d_output):
        d_weights = np.dot(d_output, self.input_data.T)
        d_biases = np.sum(d_output, axis=1, keepdims=True)
        d_input = np.dot(self.weights.T, d_output)
        return d_input, d_weights, d_biases
