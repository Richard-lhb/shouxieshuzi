import numpy as np
import skimage.measure

class PoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):
        self.input_data = input_data
        batch_size, num_channels, input_height, input_width = input_data.shape
        output_height = input_height // self.pool_size
        output_width = input_width // self.pool_size
        output = np.zeros((batch_size, num_channels, output_height, output_width))

        for b in range(batch_size):
            for c in range(num_channels):
                output[b, c] = skimage.measure.block_reduce(input_data[b, c], (self.pool_size, self.pool_size), np.max)

        return output

    def backward(self, d_output):
        batch_size, num_channels, output_height, output_width = d_output.shape
        _, _, input_height, input_width = self.input_data.shape
        d_input = np.zeros_like(self.input_data)

        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        pool_region = self.input_data[b, c, i * self.pool_size:(i + 1) * self.pool_size,
                                      j * self.pool_size:(j + 1) * self.pool_size]
                        mask = (pool_region == np.max(pool_region))
                        d_input[b, c, i * self.pool_size:(i + 1) * self.pool_size,
                        j * self.pool_size:(j + 1) * self.pool_size] = mask * d_output[b, c, i, j]

        return d_input