import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        self.in_channels = self.out_channels = self.A.shape[1]

        self.output_width = self.A.shape[2] - self.kernel + 1
        self.output_height = self.A.shape[3] - self.kernel + 1

        Z = np.zeros(shape = (self.A.shape[0], self.out_channels, self.output_width, self.output_height))
        self.Z_idx = np.zeros(shape = (self.A.shape[0], self.out_channels, self.output_width, self.output_height, 2))    # Adding 2 dimensions here for x and y axis

        argmax_axes = 2     # Here 2 denotes the last 2 axes over which argmax is computed (Not to be confused with above 2)
        filter_shape = A[:, :, 0: self.kernel, 0: self.kernel].shape
        new_shape = filter_shape[:-argmax_axes] + (np.prod(filter_shape[-argmax_axes:]),)

        for i in range(self.output_width):
            for j in range(self.output_height):
                Z[:, :, i, j] = np.max(A[:, :, i: i+self.kernel, j: j+self.kernel], axis = (2, 3))

                max_idx = A[:, :, i: i+self.kernel, j: j+self.kernel].reshape(new_shape).argmax(-1)
                idx = np.array(np.unravel_index(max_idx, filter_shape[-argmax_axes:]))

                idx[0] += i     # Adjusting the x-axis
                idx[1] += j     # Adjusting the y-axis

                self.Z_idx[:, :, i, j, 0] = idx[0]   # Assigning the x-axis
                self.Z_idx[:, :, i, j, 1] = idx[1]   # Assigning the y-axis

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(shape = (self.A.shape[0], self.in_channels, self.A.shape[2], self.A.shape[3]))

        for batch in range (self.A.shape[0]):
            for channel in range (self.in_channels):
                for i in range(self.output_width):
                    for j in range(self.output_height):
                        dLdA[batch, channel, int(self.Z_idx[batch, channel, i, j][0]), int(self.Z_idx[batch, channel, i, j][1])] += dLdZ[batch, channel, i, j]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        self.in_channels = self.out_channels = self.A.shape[1]

        self.output_width = self.A.shape[2] - self.kernel + 1
        self.output_height = self.A.shape[3] - self.kernel + 1

        Z = np.zeros(shape = (self.A.shape[0], self.out_channels, self.output_width, self.output_height))

        for i in range(self.output_width):
            for j in range(self.output_height):
                Z[:, :, i, j] = np.mean(A[:, :, i: i+self.kernel, j: j+self.kernel], axis = (2, 3))
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(shape = (self.A.shape[0], self.in_channels, self.A.shape[2], self.A.shape[3]))
        div_factor = self.kernel ** 2

        for i in range(self.output_width):
            for j in range(self.output_height):
                dLdA[:, :, i: i+self.kernel, j: j+self.kernel] += np.repeat(dLdZ[:, :, i, j, np.newaxis, np.newaxis], 
                                                                            self.kernel, axis = 2).repeat(self.kernel, axis = 3)/div_factor

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_stride1 = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA_upsampled = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA_upsampled)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_stride1 = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA_upsampled = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA_upsampled)

        return dLdA