# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        if eval:
            self.Z = Z
            self.NZ = (self.Z - self.running_M)/np.sqrt(self.running_V+self.eps)
            self.BZ = self.BW*self.NZ + self.Bb
            return self.BZ

        self.Z = Z
        self.N = Z.shape[0]
        self.num_features = Z.shape[1]

        self.M = np.mean(Z, axis = (0, 2, 3)).reshape(1, self.num_features, 1, 1)
        self.V = np.var(Z, axis = (0, 2, 3)).reshape(1, self.num_features, 1, 1)
        self.NZ = (self.Z - self.M)/np.sqrt(self.V+self.eps)
        self.BZ = self.BW*self.NZ + self.Bb

        self.running_M = (1-self.alpha)*self.M + self.alpha*self.running_M
        self.running_V = (1-self.alpha)*self.V + self.alpha*self.running_V

        return self.BZ

    def backward(self, dLdBZ):
        self.NZ = (self.Z - self.M)/np.sqrt(self.V+self.eps)
        
        self.dLdBW = np.sum(dLdBZ*self.NZ, axis = (0, 2, 3)).reshape(1, self.num_features, 1, 1)
        self.dLdBb = np.sum(dLdBZ, axis = (0, 2, 3)).reshape(1, self.num_features, 1, 1)

        dLdNZ = dLdBZ*self.BW
        dLdV = -0.5*np.sum(dLdNZ*((self.Z - self.M)/((self.V+self.eps)**1.5)), axis = (0, 2, 3)).reshape(1, self.num_features, 1, 1)

        dLdM = -1*np.sum(dLdNZ/np.sqrt(self.V+self.eps), axis = (0, 2, 3)).reshape(1, self.num_features, 1, 1)
        dLdZ = dLdNZ/np.sqrt(self.V+self.eps) + (dLdM + 2*dLdV*(self.Z - self.M))*self.num_features/self.Z.reshape(-1).shape

        return dLdZ