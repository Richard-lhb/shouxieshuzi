import numpy as np
from scipy.signal import convolve2d

class ConvLayer:
    def __init__(self, num_filters, filter_size, input_channels):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size) / (filter_size * filter_size)
        self.biases = np.zeros((num_filters, 1))

    def forward(self, input_data):
        self.input_data = input_data
        batch_size, input_channels, input_height, input_width = input_data.shape
        output_height = input_height - self.filter_size + 1
        output_width = input_width - self.filter_size + 1
        output = np.zeros((batch_size, self.num_filters, output_height, output_width))

        for b in range(batch_size):
            for f in range(self.num_filters):
                for c in range(input_channels):
                    output[b, f] += convolve2d(input_data[b, c], self.filters[f, c], mode='valid')
                output[b, f] += self.biases[f]

        return output

    def backward(self, d_output):
        batch_size, num_filters, output_height, output_width = d_output.shape
        _, input_channels, input_height, input_width = self.input_data.shape
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input_data)

        for b in range(batch_size):
            for f in range(num_filters):
                for c in range(input_channels):
                    d_filters[f, c] += convolve2d(self.input_data[b, c], d_output[b, f], mode='valid')
                    d_input[b, c] += convolve2d(d_output[b, f], np.rot90(self.filters[f, c], 2), mode='full')
                d_biases[f] += np.sum(d_output[b, f])

        return d_input, d_filters, d_biases