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
        self.W_all = None # Contains a vector with all W weight matrices
        self.U_all = None # Contains a vector will all U weight matrices.
        self.B = None
        self.C = None

        # Dynamic parameters.
        self.memory_vec = None



    def data(self):
        t = torch.tensor((2,3))
        print(t)



    def init_model(self):
        '''
        Initializes the LSTM model weights. 
        :return:
        '''
        self.W_all = torch.empty(4, self.m, self.m, dtype=torch.float64)
        self.U_all = torch.empty(4, self.m, self.m, dtype=torch.float64)

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
        if h0 is None:
            h0 = torch.empty(1, self.m, dtype=torch.float64) # Shape?

        # Initliazing the first hidden layer computaions.
        tau = X.shape[1]
        ht = h0
        ct = torch.zeros(1, self.m, dtype=torch.float64)

        ## give informative names to these torch classes
        apply_tanh = torch.nn.Tanh()
        apply_sigmoid = torch.nn.Sigmoid()
        apply_softmax = torch.nn.Softmax(dim=1)

        # create an empty tensor to store the hidden vector at each timestep
        Hs = torch.empty(h0.shape[0], X.shape[1], dtype=torch.float64)
        As = torch.empty(h0.shape[0], X.shape[1], dtype=torch.float64)
        Fs = torch.empty(self.m, tau, dtype=torch.float64) # is the 1 needed here?
        Os = torch.empty(self.m, tau, dtype=torch.float64)
        Is = torch.empty(self.m, tau, dtype=torch.float64)
        Cs = torch.empty(self.m, tau, dtype=torch.float64)
        C_Hat_s = torch.empty(self.m, tau, dtype=torch.float64)

        E = torch.zeros(4, 4, self.m, self.m)
        for i in range(4):
            E[i][i] = torch.eye(self.m)

        hprev = ht
        cprev = ct
        for t in range(tau):
            # at will have shape (4xmx1)
            at = torch.matmul(self.W_all, hprev) + torch.matmul(self.U_all, X[t])
            As[t] = at
            # Exa_t will have shape (4xmx1)
            #NOTE: Might be wrong shape.
            Fs[t] = apply_sigmoid(torch.matmul(E[0], at)) # forget gate.
            Is[t] = apply_sigmoid(torch.matmul(E[1], at)) # input gate.
            Os[t] = apply_sigmoid(torch.matmul(E[2], at)) # output gate.
            C_Hat_s[t] = apply_tanh(torch.matmul(E[3], at)) # new memory cell.
            if t < 1:
                Cs[t] = Fs[t] * cprev + Is[t] * C_Hat_s[t]
            else:
                Cs[t] = Fs[t] * Cs[t - 1] + Is[t] * C_Hat_s[t]  # final memory cell.

            Hs[t] = Os[t] * apply_tanh(Cs[t])
            hprev = Hs[t]

        # Os = torch.matmul(Hs, self.W_o) + self.C
        P = apply_softmax(Os)

        # compute the loss
        loss = torch.mean(-torch.log(P[np.arange(tau), y]))  # use this line if storing inputs row-wise
        return loss

    def backward(self, loss):
        loss.backward()
        return (self.W_all.grad, self.U_all.grad) # unsure if correct.
         
