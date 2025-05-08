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



    def data(self):
        t = torch.tensor((2,3))
        print(t)



    def init_model(self):
        return



    def fit(self):
        return

