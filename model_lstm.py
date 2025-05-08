'''
This file contains the LSTM model.
'''

import torch
import numpy as np
from DataProcessing import *
import copy

class LSTM:
    def __init__(self, X, m=100, n_layers=2, seq_len=25, lr=0.01, lam=0):

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

        # The length of each sequence being inputted to the backprop.
        self.seq_len = seq_len

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
        :param X: the BATCH encoded input vector.
        :param y: the BATCH encoded target vector, NOT one-hot-encoded.
        :return: loss
        '''
        if h0 is None:
            h0 = torch.empty(self.m, 1, dtype=torch.float64) # shape (m, 1).

        X = torch.from_numpy(X)

        assert(X.shape[0] == self.seq_len, f"X shape: {X.shape} != seq_len:{self.seq_len}")  # for catching errors.
        tau = self.seq_len

        apply_tanh = torch.nn.Tanh()
        apply_sigmoid = torch.nn.Sigmoid()
        apply_softmax = torch.nn.Softmax(dim=1)

        As, Fs, Is, Os, Cs, C_Hat_s, Hs = [], [], [], [], [], [], []
        hprev = h0
        cprev = torch.zeros(1, self.m, dtype=torch.float64)
        for t in range(tau):
            # at will have shape (4xmx1)
            at = torch.matmul(self.W_all, hprev) + torch.matmul(self.U_all, X[t].reshape(X[t].shape[0], 1))
            As.append(at.squeeze())
            # Exa_t will have shape (4xmx1)
            #NOTE: Might be wrong shape.

            Fs.append(apply_sigmoid(at[0]).squeeze()) # forget gate.
            Is.append(apply_sigmoid(at[1]).squeeze()) # input gate.
            Os.append(apply_sigmoid(at[2]).squeeze()) # output gate.
            C_Hat_s.append(apply_tanh(at[3]).squeeze()) # new memory cell.
            if t < 1:
                Cs.append(Fs[t] * cprev + Is[t] * C_Hat_s[t])
            else:
                Cs.append(Fs[t] * Cs[t - 1] + Is[t] * C_Hat_s[t]) # final memory cell.

            Hs.append(Os[t] * apply_tanh(Cs[t]))
            hprev = Hs[t].reshape(self.m, 1)

        # Os = torch.matmul(Hs, self.W_o) + self.C
        Hs = torch.stack(Hs, dim=0)  # shape (tau, m, 1)
        As = torch.stack(As, dim=0)
        Fs = torch.stack(Fs, dim=0)
        Is = torch.stack(Is, dim=0)
        Os = torch.stack(Os, dim=0)
        C_Hat_s = torch.stack(C_Hat_s, dim=0)
        Cs = torch.stack(Cs, dim=0)
        P = apply_softmax(Os)

        # compute the loss
        loss = torch.mean(-torch.log(P[np.arange(tau), y]))  # use this line if storing inputs row-wise
        return loss



    def backward(self, loss):
        loss.backward()
        return (self.W_all.grad, self.U_all.grad) # unsure if correct.



    def train(self):

        return


data, unique_chars = ReadData()
char_to_ind, ind_to_char = GetDicts(unique_chars)
X = ConvertToOneHot(data, char_to_ind)

X_seq = X[0:25]
y_seq = data[1:26] # not one-hot encoded.
y_seq_indices = [char_to_ind[char] for char in y_seq]

lstm = LSTM(X)

loss = lstm.forward(X_seq, y_seq_indices)
grads_W, grads_U = lstm.backward(loss)
print(grads_W[0])

