from layers.conv_layer import ConvLayer
from layers.pooling_layer import PoolingLayer
from layers.fully_connected_layer import FullyConnectedLayer
from utils.activation_functions import sigmoid, sigmoid_derivative
import numpy as np


class LeNet:
    def __init__(self):
        self.conv1 = ConvLayer(num_filters=6, filter_size=5, input_channels=1)
        self.pool1 = PoolingLayer(pool_size=2)
        self.conv2 = ConvLayer(num_filters=16, filter_size=5, input_channels=6)
        self.pool2 = PoolingLayer(pool_size=2)
        self.fc1 = FullyConnectedLayer(input_size=16 * 4 * 4, output_size=120)
        self.fc2 = FullyConnectedLayer(input_size=120, output_size=84)
        self.fc3 = FullyConnectedLayer(input_size=84, output_size=10)

    def forward(self, input_data):
        self.input_data = input_data
        # Conv1 + Sigmoid + Pool1
        conv1_out = self.conv1.forward(input_data)
        self.sigmoid1_input = conv1_out
        sigmoid1_out = sigmoid(conv1_out)
        pool1_out = self.pool1.forward(sigmoid1_out)

        # Conv2 + Sigmoid + Pool2
        conv2_out = self.conv2.forward(pool1_out)
        self.sigmoid2_input = conv2_out
        sigmoid2_out = sigmoid(conv2_out)
        pool2_out = self.pool2.forward(sigmoid2_out)
        self.pool2_out_shape = pool2_out.shape

        # Flatten
        flattened = pool2_out.reshape((pool2_out.shape[0], -1)).T

        # FC1 + Sigmoid
        fc1_out = self.fc1.forward(flattened)
        self.sigmoid3_input = fc1_out
        sigmoid3_out = sigmoid(fc1_out)

        # FC2 + Sigmoid
        fc2_out = self.fc2.forward(sigmoid3_out)
        self.sigmoid4_input = fc2_out
        sigmoid4_out = sigmoid(fc2_out)

        # FC3 (No activation for output layer)
        fc3_out = self.fc3.forward(sigmoid4_out)
        return fc3_out

    def backward(self, d_output):
        # FC3 backward
        d_fc3_input, d_fc3_weights, d_fc3_biases = self.fc3.backward(d_output)

        # FC2 backward
        d_sigmoid4 = sigmoid_derivative(self.sigmoid4_input) * d_fc3_input
        d_fc2_input, d_fc2_weights, d_fc2_biases = self.fc2.backward(d_sigmoid4)

        # FC1 backward
        d_sigmoid3 = sigmoid_derivative(self.sigmoid3_input) * d_fc2_input
        d_fc1_input, d_fc1_weights, d_fc1_biases = self.fc1.backward(d_sigmoid3)

        # Reshape
        d_flattened = d_fc1_input.reshape(self.pool2_out_shape)

        # Pool2 backward
        d_pool2 = self.pool2.backward(d_flattened)
        d_sigmoid2 = sigmoid_derivative(self.sigmoid2_input) * d_pool2
        d_conv2_input, d_conv2_filters, d_conv2_biases = self.conv2.backward(d_sigmoid2)

        # Pool1 backward
        d_pool1 = self.pool1.backward(d_conv2_input)
        d_sigmoid1 = sigmoid_derivative(self.sigmoid1_input) * d_pool1
        d_conv1_input, d_conv1_filters, d_conv1_biases = self.conv1.backward(d_sigmoid1)

        return d_conv1_filters, d_conv1_biases, d_conv2_filters, d_conv2_biases, \
            d_fc1_weights, d_fc1_biases, d_fc2_weights, d_fc2_biases, \
            d_fc3_weights, d_fc3_biases

    def update_weights(self, d_conv1_filters, d_conv1_biases, d_conv2_filters, d_conv2_biases,
                       d_fc1_weights, d_fc1_biases, d_fc2_weights, d_fc2_biases,
                       d_fc3_weights, d_fc3_biases, learning_rate):
        # Apply gradient clipping
        for grad in [d_conv1_filters, d_conv1_biases, d_conv2_filters, d_conv2_biases,
                     d_fc1_weights, d_fc1_biases, d_fc2_weights, d_fc2_biases,
                     d_fc3_weights, d_fc3_biases]:
            np.clip(grad, -1, 1, out=grad)

        # Update weights
        self.conv1.filters -= learning_rate * d_conv1_filters
        self.conv1.biases -= learning_rate * d_conv1_biases
        self.conv2.filters -= learning_rate * d_conv2_filters
        self.conv2.biases -= learning_rate * d_conv2_biases
        self.fc1.weights -= learning_rate * d_fc1_weights
        self.fc1.biases -= learning_rate * d_fc1_biases
        self.fc2.weights -= learning_rate * d_fc2_weights
        self.fc2.biases -= learning_rate * d_fc2_biases
        self.fc3.weights -= learning_rate * d_fc3_weights
        self.fc3.biases -= learning_rate * d_fc3_biases