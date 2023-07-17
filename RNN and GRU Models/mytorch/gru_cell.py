import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.n_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        self.r = self.r_act.forward(np.matmul(self.Wrx, x) + self.brx + np.matmul(self.Wrh, self.hidden) + self.brh)   # Dimensions = (hidden_dim)
        self.z = self.z_act.forward(np.matmul(self.Wzx, x) + self.bzx + np.matmul(self.Wzh, self.hidden) + self.bzh)   # Dimensions = (hidden_dim)
        self.n = self.n_act.forward(np.matmul(self.Wnx, x) + self.bnx + self.r*(np.matmul(self.Wnh, self.hidden) + self.bnh))  # Dimensions = (hidden_dim)
        h_t    = (1-self.z)*self.n + self.z*self.hidden     # Dimensions = (hidden_dim)
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly

        x_reshaped = self.x.reshape(-1, 1)                     # Dimesnions = (input_dim, 1)
        hidden_reshaped = self.hidden.reshape(-1, 1)           # Dimesnions = (hidden_dim, 1)

        r_reshaped = self.r.reshape(-1, 1)                     # Dimesnions = (hidden_dim, 1)
        n_reshaped = self.n.reshape(-1, 1)                     # Dimesnions = (hidden_dim, 1)
        z_reshaped = self.z.reshape(-1, 1)                     # Dimesnions = (hidden_dim, 1)

        delta_reshaped = delta.reshape(-1, 1)                  # Dimesnions = (hidden_dim, 1)

        # Delta = dLdht

        dLdz = delta_reshaped*(hidden_reshaped - n_reshaped)    # Dimesnions = (hidden_dim, 1)
        dLdn = delta_reshaped*(1-z_reshaped)                    # Dimesnions = (hidden_dim, 1)
        dLdr = dLdn*self.n_act.backward(state = n_reshaped)*(np.matmul(self.Wnh, hidden_reshaped) + self.bnh.reshape(self.h, 1)) # Chain Rule # Dimesnions = (hidden_dim, 1)

        ###############################################################################
        ###############################################################################
        dLdn_back = dLdn*self.n_act.backward().reshape(self.h, 1)

        self.dWnx += np.matmul(dLdn_back, x_reshaped.T)
        self.dbnx += (dLdn_back).reshape(self.h)

        self.dWnh += np.matmul(dLdn_back*r_reshaped, hidden_reshaped.T)
        self.dbnh += (dLdn_back*r_reshaped).reshape(self.h)
        ###############################################################################
        ###############################################################################


        ###############################################################################
        ###############################################################################        
        dLdz_back = dLdz*self.z_act.backward().reshape(self.h, 1)

        self.dWzx += np.matmul(dLdz_back, x_reshaped.T)
        self.dbzx += (dLdz_back).reshape(self.h)

        self.dWzh += np.matmul(dLdz_back, hidden_reshaped.T)
        self.dbzh += (dLdz_back).reshape(self.h)
        ###############################################################################
        ###############################################################################


        ###############################################################################
        ###############################################################################
        dLdr_back = dLdr*self.r_act.backward().reshape(self.h, 1)

        self.dWrx += np.matmul(dLdr_back, x_reshaped.T)
        self.dbrx += (dLdr_back).reshape(self.h)

        self.dWrh += np.matmul(dLdr_back, hidden_reshaped.T)
        self.dbrh += (dLdr_back).reshape(self.h)
        ###############################################################################
        ###############################################################################


        dh_prev_t = (delta_reshaped*z_reshaped).T + np.matmul((dLdn_back*r_reshaped).T, self.Wnh) + np.matmul(dLdz_back.T, self.Wzh) + np.matmul(dLdr_back.T, self.Wrh)
        dx = np.matmul(dLdn_back.T, self.Wnx) + np.matmul(dLdz_back.T, self.Wzx) + np.matmul(dLdr_back.T, self.Wrx)
        
        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
