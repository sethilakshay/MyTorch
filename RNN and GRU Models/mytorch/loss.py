# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss

    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        TODO: Implement this function similar to how you did for HW1P1 or HW2P1.
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """

        self.logits = x
        self.labels = y

        N = self.logits.shape[1]

        self.softmax = np.exp(self.logits)/np.sum(np.exp(self.logits), axis = 1).reshape(-1, 1)
        self.loss = np.sum(-1*self.labels*np.log(self.softmax), axis=1)/self.logits.shape[1]

        return self.loss

    def backward(self):
        """
        TODO: Implement this function similar to how you did for HW1P1 or HW2P1.
        Return:
            out (np.array): (batch size, 10)
        """

        self.gradient = self.softmax - self.labels

        return self.gradient
