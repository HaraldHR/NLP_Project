'''
This file contains the LSTM model.
'''

import torch
import numpy as np

class LSTM:
    def __init__(self, m=100, n_layers=2, lr=0.01, lam=0):

        # The starting learning rate of the model.
        self.lr = lr

        # The regularization term.
        self.lam = lam

        # The number of nodes in each layer.
        self.m = m

        # The number of hidden layers in the model.
        self.L = n_layers

        # The dimension of the input and output.
        self.K = None

        # Initializing trainable parameters
        self.V = None
        self.W = None
        self.U = None
        self.B = None
        self.C = None

        # Dynamic parameters.
        self.memory_vec = None



    def data(self):
        t = torch.tensor((2,3))
        print(t)



    def init_model(self):
        return



    def forward(self, X, y, h0=None):
        '''
        Computes the forward pass of the LSTM model to make a prediction.
        :param X: the encoded input vector.
        :param y: the encoded target vector.
        :return:
        '''
        if h0 is None:
            h0 = torch.empty(1, self.m, dtype=torch.float64) # Shape?


        # Initliazing the first hidden layer computaions.
        tau = X.shape[1]
        ht = h0

        ## give informative names to these torch classes
        apply_tanh = torch.nn.Tanh()
        apply_softmax = torch.nn.Softmax(dim=1)

        # create an empty tensor to store the hidden vector at each timestep
        Hs = torch.empty(h0.shape[0], X.shape[1], dtype=torch.float64)
        As = torch.empty(h0.shape[0], X.shape[1], dtype=torch.float64)

        hprev = ht
        for t in range(tau):
            at = torch.matmul(self.W, hprev) + torch.matmul(self.U, X[:, t:t + 1]) + \
                 self.B
            As[:, t:t + 1] = at
            ht = apply_tanh(at)
            Hs[:, t:t + 1] = ht
            hprev = ht

        Os = torch.matmul(Hs, self.V) + self.C
        P = apply_softmax(Os)

        # compute the loss
        loss = torch.mean(-torch.log(P[np.arange(tau), y]))  # use this line if storing inputs row-wise
        return loss



