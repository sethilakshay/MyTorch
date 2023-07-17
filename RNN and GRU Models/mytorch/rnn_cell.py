import numpy as np
from activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """

        """
        ht = tanh(Wihxt + bih + Whhht−1 + bhh) 

        W_ih = (hidden_size, input_size)
        W_hh = (hidden_size, hidden_size)
        b_ih, b_hh = (hidden_size,)
        """

        # Matrix multiplication done in a way to align dimensions
        Z_t = np.matmul(x, self.W_ih.T) + self.b_ih + (np.matmul(h_prev_t, self.W_hh.T) + self.b_hh) # In brackets is the hidden part
        h_t = self.activation.forward(Z_t)

        return h_t #Final Dimension = (batch_size, hidden_size)

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input (see writeup for explanation)
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h_t: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh_prev_t: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        batch_size = delta.shape[0]
        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.

        dz = delta*self.activation.backward(h_t)  #Shape = (batch_size, hidden_size)

        # 1) Compute the averaged gradients of the weights and biases (Use Dimensionality Analysis to match the input and output dimensions)
        self.dW_ih += np.matmul(dz.T, h_prev_l)/batch_size  # (hidden_size, input_size)
        self.dW_hh += np.matmul(dz.T, h_prev_t)/batch_size  # (hidden_size, hidden_size)
        self.db_ih += np.sum(dz, axis = 0)/batch_size
        self.db_hh += np.sum(dz, axis = 0)/batch_size

        # # 2) Compute dx, dh_prev_t
        dx        = np.matmul(dz, self.W_ih)
        dh_prev_t = np.matmul(dz, self.W_hh)

        # 3) Return dx, dh_prev_t
        return dx, dh_prev_t
