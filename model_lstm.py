'''
This file contains the LSTM model.
'''

import torch
import numpy as np
from DataProcessing import *
import copy

class LSTM:
    def __init__(self, X, test_size=0.2, m=[100, 50], n_layers=2, seq_len=25, lr=0.01, lam=0):

        # The one-hot encoded data
        self.X = X
        self.test_size = test_size

        # The starting learning rate of the model.
        self.lr = lr

        # The regularization term.
        self.lam = lam

        # The number of nodes in each layer.
        self.m = m

        # The number of hidden layers in the model.
        self.L = n_layers

        # The dimension of the input and output.
        self.K = [X.shape[1]]

        # The length of each sequence being inputted to the backprop.
        self.seq_len = seq_len

        # Initializing trainable parameters
        self.W_all = {} # Contains a vector with all W weight matrices
        self.U_all = {} # Contains a vector will all U weight matrices.
        self.V = None # output weight matrix
        self.B = {}
        self.C = None

        # Dynamic parameters.
        self.memory_vec = None
        
        self.init_model()



    def init_model(self):
        '''
        Initializes the LSTM model weights. 
        :return:
        '''
        for l in range(self.L): # K[0] is already adde to the list earlier.
            self.K.append(self.m[l]) # K_l = m_l - 1

        # Xavier initialization for layer 1 weights.
        for l in range(self.L):
            W_all, U_all, B = self.init_weights(self.m[l], self.K[l])
            self.W_all[l] = torch.tensor(W_all, requires_grad=True)
            self.U_all[l] = torch.tensor(U_all, requires_grad=True)
            self.B[l] = torch.tensor(B, requires_grad=True)
        
        self.V = torch.empty(self.K[0], self.m[-1], dtype=torch.float64, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.V)
        self.C = torch.empty(self.K[0], dtype=torch.float64, requires_grad=True)



    def init_weights(self, m, K):
            W_all = torch.empty(4, m, m, dtype=torch.float64)
            U_all = torch.empty(4, m, K, dtype=torch.float64)

            B = torch.zeros(4, m, dtype=torch.float64) # Gate biases
            for i in range(4):
                torch.nn.init.xavier_uniform_(W_all[i])
                torch.nn.init.xavier_uniform_(U_all[i])

            return W_all, U_all, B
        

    def forward(self, X, y, h0=None):
        '''
        Computes the forward pass of the LSTM model to make a prediction.
        :param X: the BATCH encoded input vector.
        :param y: the BATCH encoded target vector, NOT one-hot-encoded.
        :return: loss
        '''
        #print(self.W_all[0][0])
        #print(self.U_all[0][0])
        #print(self.B[0])
        if h0 is None:
            h0 = torch.empty(self.m[0], 1, dtype=torch.float64) # shape (m, 1).

        X = torch.from_numpy(X)

        assert X.shape[0] == self.seq_len, f"X shape: {X.shape} != seq_len:{self.seq_len}"  # for catching errors.
        tau = self.seq_len

        apply_tanh = torch.nn.Tanh()
        apply_sigmoid = torch.nn.Sigmoid()
        apply_softmax = torch.nn.Softmax(dim=1)

        

        hprev = h0
        cprev = torch.zeros(1, self.m[0], dtype=torch.float64)

        
        Hs_L = torch.zeros(tau, self.m[self.L - 1], dtype=torch.float64) # the last Hs, which we use for s and P calculation.
        current_X = X
        for l in range(self.L): # for each layer
            Hs = torch.zeros(tau, self.m[l], dtype=torch.float64) # Hs for this layer
            for t in range(tau):
                # at will have shape (4 x m x 1)
                

                at = torch.matmul(self.W_all[l], hprev) + torch.matmul(self.U_all[l], current_X[t].reshape(current_X[t].shape[0], 1)) + self.B[l].unsqueeze(2) # Include biases
                #As[l].append(at.squeeze())
                # Exa_t will have shape (4xmx1)
                #NOTE: Might be wrong shape.

                f_t = apply_sigmoid(at[0]).squeeze() # forget gate.
                i_t = apply_sigmoid(at[1]).squeeze() # input gate.
                o_t = apply_sigmoid(at[2]).squeeze() # output gate.
                c_hat_t = apply_tanh(at[3]).squeeze() # new memory cell.
                cprev = f_t * cprev + i_t * c_hat_t

                h_t = o_t * apply_tanh(cprev)
                Hs[t] = h_t
                
                if l == self.L - 1: Hs_L = Hs[t]
                
                hprev = h_t.reshape(self.m[l], 1)
                

            # input for next layer:
            current_X = Hs.squeeze()
            # the first values for the next layer:
            if l < self.L - 1: 
                hprev = torch.empty(self.m[l + 1], 1, dtype=torch.float64) # new h0
                cprev = torch.zeros(1, self.m[l + 1], dtype=torch.float64) # new c0
            #print(current_X.squeeze().shape)
            #print(hprev.shape)
            #print(cprev.shape)
            print("Layer " + str(l + 1) + " complete!")
            

        s = torch.matmul(self.V, Hs_L.reshape(self.m[-1], 1)) + self.C
        #print(Hs[-1][-1].reshape(self.m[-1], 1).shape)
        P = apply_softmax(s).squeeze()

        # compute the loss
        loss = torch.mean(-torch.log(P[np.arange(tau), y]))  # use this line if storing inputs row-wise 
        return loss



    def backward(self, loss):
        loss.backward()

    
    def synth_text(self, x0, n, ind_to_char, char_to_ind, rng, h0 = None):
        '''
        Computes the forward pass of the LSTM model to make a prediction.
        :param x0: the first character
        :param n: length of the synthesized text
        :param rng: the previously initialized rng
        :return:
        '''

        if h0 is None:
            h0 = torch.empty(self.m, 1, dtype=torch.float64) # Shape?

        # Initliazing the first hidden layer computaions.
        ht = h0
        ct = torch.zeros(1, self.m, dtype=torch.float64)

        ## give informative names to these torch classes
        apply_tanh = torch.nn.Tanh()
        apply_sigmoid = torch.nn.Sigmoid()
        apply_softmax = torch.nn.Softmax(dim = 0)

        hprev = ht
        cprev = ct
        x = torch.zeros(1, self.K, dtype=torch.float64)
        x[0, char_to_ind[x0]] = 1
        synth_text = x0
        for t in range(n):
            # at will have shape (4xmx1)
            at = torch.matmul(self.W_all, hprev) + torch.matmul(self.U_all, x.reshape(x.shape[1], 1)) + self.B.unsqueeze(2)
            #As.append(at.squeeze())
            # Exa_t will have shape (4xmx1)
            #NOTE: Might be wrong shape.    

            f_t = apply_sigmoid(at[0]).squeeze() # forget gate.
            i_t = apply_sigmoid(at[1]).squeeze() # input gate.
            o_t = apply_sigmoid(at[2]).squeeze() # output gate.
            c_hat_t = apply_tanh(at[3]).squeeze() # new memory cell.
            
            cprev = f_t * cprev + i_t * c_hat_t
            hprev = (o_t * apply_tanh(cprev)).reshape(self.m, 1)

            s_t = torch.matmul(self.V, hprev.reshape(self.m, 1)) + self.C.unsqueeze(1)
            p_t = apply_softmax(s_t)

            # sample (randomly from prob. dist. p_t) and add to synthesized text:
            cp = np.cumsum(p_t.detach().numpy(), axis=0)
            a = rng.uniform(size=1)
            ii = np.argmax(cp - a > 0) 

            x = torch.zeros(1, self.K, dtype=torch.float64)
            x[0, ii] = 1
            synth_text += ind_to_char[ii]

        return synth_text

    def fit(self, epochs=5):
        n_batches = len(self.X) // self.seq_len  # number of full batches
        trimmed_len = n_batches * self.seq_len
        X_trimmed = X[:trimmed_len]  # trim off the remainder, OBS: Losing some characters, maybe not necessary!
        Y_trimmed = X[1:trimmed_len + 1]
        batches_X = X_trimmed.reshape(n_batches, self.seq_len, * X.shape[1:])
        batches_Y = Y_trimmed.reshape(n_batches, self.seq_len, * X.shape[1:])

        assert np.allclose(batches_Y[0, 0], batches_X[0, 1]), f"Data and labels are shifted wrong: Y: {batches_Y[0, 0]} != X: {batches_X[0, 1]}"

        print(batches_X.shape)
        for _ in range(epochs):
            for i in range(batches_X.shape[0]):
                loss = self.forward(batches_X[i], batches_Y[i])
                grads_W, grads_U = self.backward(loss)

                # Updates the weights
                self.W_all -= grads_W * self.lr
                self.U_all -= grads_U * self.lr




rng = np.random.default_rng()
# get the BitGenerator used by default
BitGen = type(rng.bit_generator)
# use the state from a fresh bit generator
seed = 1
rng.bit_generator.state = BitGen(seed).state


data, unique_chars = ReadData()
char_to_ind, ind_to_char = GetDicts(unique_chars)
X = ConvertToOneHot(data, char_to_ind)

X_seq = X[0:25]
y_seq = data[1:26] # not one-hot encoded.
y_seq_indices = [char_to_ind[char] for char in y_seq]

lstm = LSTM(X, m=[100, 50, 40], n_layers=3)

#print(lstm.W_all)
loss = lstm.forward(X_seq, y_seq_indices)
grads = lstm.backward(loss)
#print(lstm.C.grad)
print(lstm.W_all[2].grad[0])
#print(torch.any(lstm.U_all[0].grad != 0))

#synth_text = lstm.synth_text("a", 25, ind_to_char, char_to_ind, rng)
#print(synth_text)
#lstm.fit()

