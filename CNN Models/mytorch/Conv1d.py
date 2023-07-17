# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
            W (np.array): (out_channels, in_channels, kernel_size)
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        self.input_size = self.A.shape[2]
        self.output_size = self.input_size - self.kernel_size + 1
        
        Z = np.zeros(shape = (self.A.shape[0], self.out_channels, self.output_size))

        # Computing Z using tensordot
        for i in range(self.output_size):    # Iterating over the output size
            Z[:, :, i] = np.tensordot(self.A[:, :, i: i+self.kernel_size], self.W, axes = ((1, 2), (1, 2))) + self.b

        return Z

    def backward(self, dLdZ):
        """
            W (np.array): (out_channels, in_channels, kernel_size)
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Computing dLdW
        for i in range(self.kernel_size):
            self.dLdW[:, :, i] = np.tensordot(dLdZ, self.A[:, :, i: i+self.output_size], axes = ((0, 2), (0, 2))) 

        # Computing dLdb
        self.dLdb = np.sum(dLdZ, axis = (0, 2)) 


        dLdA = np.zeros(shape = self.A.shape)
        dLdZ_Padded = np.pad(dLdZ, pad_width = ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1)), mode = 'constant', constant_values = 0)
        # Computing dLdA
        for i in range(self.input_size):
            dLdA[:, :, i] = np.tensordot(dLdZ_Padded[:, :, i: i+self.kernel_size], self.W[:, :, ::-1], axes = ((1, 2), (0, 2)))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(self.stride)

    def forward(self, A):
        """
            W (np.array): (out_channels, in_channels, kernel_size)
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Call Conv1d_stride1
        Z_Stride1 = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z_Stride1)

        return Z

    def backward(self, dLdZ):
        """
            
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdA_downsampled = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdA_downsampled)
        
        return dLdA