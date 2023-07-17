import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
            W (np.array): (out_channels, in_channels, kernel_size, kernel_size)
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        self.input_height = self.A.shape[2]
        self.input_width = self.A.shape[3]

        self.output_height = self.input_height - self.kernel_size + 1
        self.output_width = self.input_width - self.kernel_size + 1

        Z = np.zeros(shape = (A.shape[0], self.out_channels, self.output_height, self.output_width))

        # Computing Z using tensordot
        for i in range(self.output_height):
            for j in range(self.output_width):
                Z[:, :, i, j] = np.tensordot(self.A[:, :, i: i+self.kernel_size, j: j+self.kernel_size], self.W, axes = ((1, 2, 3), (1, 2, 3))) + self.b

        return Z

    def backward(self, dLdZ):
        """
            W (np.array): (out_channels, in_channels, kernel_size, kernel_size)
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Computing dLdW
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.dLdW[:, :, i, j] = np.tensordot(dLdZ, self.A[:, :, i: i+self.output_height, j: j+self.output_width], axes = ((0, 2, 3), (0, 2, 3))) 

        self.dLdb = np.sum(dLdZ, axis = (0, 2, 3))

        dLdA = np.zeros(shape = self.A.shape)
        dLdZ_Padded = np.pad(dLdZ, pad_width = ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1), 
                                                (self.kernel_size-1, self.kernel_size-1)), mode = 'constant', constant_values = 0)
        # Computing dLdA
        for i in range(self.input_height):
            for j in range(self.input_width):
                dLdA[:, :, i, j] = np.tensordot(dLdZ_Padded[:, :, i: i+self.kernel_size, j: j+self.kernel_size], 
                                             self.W[:, :, ::-1, ::-1], axes = ((1, 2, 3), (0, 2, 3)))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
            W (np.array): (out_channels, in_channels, kernel_size, kernel_size)
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Call Conv2d_stride1
        Z_Stride1 = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z_Stride1)

        return Z

    def backward(self, dLdZ):
        """
            W (np.array): (out_channels, in_channels, kernel_size, kernel_size)
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample1d backward
        dLdA_downsampled = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA_downsampled)

        return dLdA