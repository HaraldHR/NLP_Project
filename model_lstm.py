'''
This file contains the LSTM model.
'''

import torch
import numpy as np
from DataProcessing import *

class LSTM:
    def __init__(self, X, m=100, n_layers=2, lr=0.01, lam=0):

        # The one-hot encoded data
        self.X = X

        # The starting learning rate of the model.
        self.lr = lr

        # The regularization term.
        self.lam = lam

        # The number of nodes in each layer.
        self.m = m

        # The number of hidden layers in the model.
        self.L = n_layers

        # The dimension of the input and output.
        self.K = X.shape[1]

        # Initializing trainable parameters
        self.W_all = None # Contains a vector with all W weight matrices
        self.U_all = None # Contains a vector will all U weight matrices.
        self.B = None
        self.C = None

        # Dynamic parameters.
        self.memory_vec = None
        
        self.init_model()



    def data(self):
        t = torch.tensor((2,3))
        print(t)



    def init_model(self):
        '''
        Initializes the LSTM model weights. 
        :return:
        '''
        self.W_all = torch.empty(4, self.m, self.m, dtype=torch.float64, requires_grad=True)
        self.U_all = torch.empty(4, self.m, self.K, dtype=torch.float64, requires_grad=True)

        # Xavier initialization for all weights.
        for i in range(4):
            torch.nn.init.xavier_uniform_(self.W_all[i])
            torch.nn.init.xavier_uniform_(self.U_all[i])



    def forward(self, X, y, h0=None):
        '''
        Computes the forward pass of the LSTM model to make a prediction.
        :param X: the encoded input vector.
        :param y: the encoded target vector.
        :return:
        '''

        X = torch.from_numpy(X)
        if h0 is None:
            h0 = torch.empty(self.m, 1, dtype=torch.float64) # Shape?

        # Initliazing the first hidden layer computaions.
        tau = X.shape[1]
        ht = h0
        ct = torch.zeros(1, self.m, dtype=torch.float64)

        ## give informative names to these torch classes
        apply_tanh = torch.nn.Tanh()
        apply_sigmoid = torch.nn.Sigmoid()
        apply_softmax = torch.nn.Softmax(dim=1)

        # create an empty tensor to store the hidden vector at each timestep
        Hs = torch.empty(tau, self.m, dtype=torch.float64)
        As = torch.empty(tau, 4, self.m, dtype=torch.float64)
        Fs = torch.empty(tau, self.m, dtype=torch.float64) # is the 1 needed here?
        Os = torch.empty(tau, self.m, dtype=torch.float64)
        Is = torch.empty(tau, self.m, dtype=torch.float64)
        Cs = torch.empty(tau, self.m, dtype=torch.float64)
        C_Hat_s = torch.empty(tau, self.m, dtype=torch.float64)

        E = torch.zeros(4, 4, self.m, self.m, dtype=torch.float64)
        for i in range(4):
            E[i][i] = torch.eye(self.m)

        hprev = ht
        cprev = ct
        for t in range(tau):
            # at will have shape (4xmx1)
            at = torch.matmul(self.W_all, hprev) + torch.matmul(self.U_all, X[t].reshape(X[t].shape[0], 1))
            As[t] = at.reshape(at.shape[0], at.shape[1])
            # Exa_t will have shape (4xmx1)
            #NOTE: Might be wrong shape.

            Fs[t] = apply_sigmoid(at[0]).reshape(1, self.m) # forget gate.
            Is[t] = apply_sigmoid(at[1]).reshape(1, self.m) # input gate.
            Os[t] = apply_sigmoid(at[2]).reshape(1, self.m) # output gate.
            C_Hat_s[t] = apply_tanh(at[3]).reshape(1, self.m) # new memory cell.
            print("TEST")
            if t < 1:
                Cs[t] = Fs[t] * cprev + Is[t] * C_Hat_s[t]
            else:
                Cs[t] = Fs[t] * Cs[t - 1] + Is[t] * C_Hat_s[t]  # final memory cell.

            Hs[t] = Os[t] * apply_tanh(Cs[t])
            hprev = Hs[t].reshape(self.m, 1)

        # Os = torch.matmul(Hs, self.W_o) + self.C
        P = apply_softmax(Os)

        # compute the loss
        loss = torch.mean(-torch.log(P[np.arange(tau), y]))  # use this line if storing inputs row-wise
        return loss

    def backward(self, loss):
        loss.backward()
        return (self.W_all.grad, self.U_all.grad) # unsure if correct.
         


data, unique_chars = ReadData()
char_to_ind, ind_to_char = GetDicts(unique_chars)
X = ConvertToOneHot(data, char_to_ind)

X_seq = X[0:25]
y_seq = X[1:26]

network = LSTM(X)

network.forward(X_seq, y_seq)
