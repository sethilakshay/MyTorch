# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels = 24, out_channels = 8, kernel_size = 8, stride = 4)
        self.conv2 = Conv1d(in_channels = 8, out_channels = 16, kernel_size = 1, stride = 1)
        self.conv3 = Conv1d(in_channels = 16, out_channels = 4, kernel_size = 1, stride = 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        """
            w.shape  = (input_shape, out_channels) where input_shape = kernel_size*in_channels
            w1.shape = (192, 8)
            w2.shape = (8, 16)
            w3.shape = (16, 4)

            ******Desired Shape******
            W (np.array): (out_channels, in_channels, kernel_size)
        """
        w1, w2, w3 = weights
        self.conv1.conv1d_stride1.W = np.transpose(w1.T.reshape(8, 8, 24), axes = (0, 2, 1))
        self.conv2.conv1d_stride1.W = np.transpose(w2.T.reshape(16, 1, 8), axes = (0, 2, 1))
        self.conv3.conv1d_stride1.W = np.transpose(w3.T.reshape(4, 1, 16), axes = (0, 2, 1))

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        """
            The input is of 8 size, let us say it looks like -------- (8 dashes) and has to scanned with a stride of 4
            To distributed scan should be done amongst three layers

            => Layer 1 will only look at 2 inputs -- (2 dashesh) and will stride with 2 (as total stride is 4)
            => Layer 2 will look at 2 outputs of Layer 1 and will effectively look at 4 inputs => -- -- (as 1 input of Layer 1 = -- (2 dashes)) 
                    It will stride again with 2 as total stride is 4
            => Layer 3 has to look at complete input => 8 dashes therefore it will look at 2 outputs of Layer 2 which in turn look at 2 outputs of Layer 1
                    which in effect looks at 2 inputs
                    Layer 3 will therefore look at ---- ---- It will stride with 1 only as previous combined have strided with 4
        """
        self.conv1 = Conv1d(in_channels = 24, out_channels = 2, kernel_size = 2, stride = 2)
        self.conv2 = Conv1d(in_channels = 2, out_channels = 8, kernel_size = 2, stride = 2)
        self.conv3 = Conv1d(in_channels = 8, out_channels = 4, kernel_size = 2, stride = 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        """
            w.shape  = (input_shape, out_channels) where input_shape = kernel_size*in_channels
            w1.shape = (192, 8)
            w2.shape = (8, 16)
            w3.shape = (16, 4)

            ******Desired Shape******
            W (np.array): (out_channels, in_channels, kernel_size)
        """
        w1, w2, w3 = weights

        self.conv1.conv1d_stride1.W = np.transpose(w1.T[:2, :48].reshape(2, 2, 24), axes = (0, 2, 1))   #w1.T sliced to align the dimensions with above logic
        self.conv2.conv1d_stride1.W = np.transpose(w2.T[:8, :4].reshape(8, 2, 2), axes = (0, 2, 1))     #w2.T sliced to align the dimensions with above logic
        self.conv3.conv1d_stride1.W = np.transpose(w3.T.reshape(4, 2, 8), axes = (0, 2, 1))

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
