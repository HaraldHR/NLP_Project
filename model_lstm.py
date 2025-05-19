'''
This file contains the LSTM model.
'''

import torch
import numpy as np
from DataProcessing import *
import copy
from tqdm import tqdm
from collections import defaultdict

class LSTM:
    def __init__(self, X, test_size=0.2, m=[100, 50], n_layers=2, seq_len=25, lr=0.01, lam=0):

        # The one-hot encoded data.
        self.X = X

        # The test/train split.
        self.test_size = test_size

        # Mappings.
        self.char2ind = defaultdict(int) # Default dict creates a new assignment (int = 1) if key does not exist.
        self.ind2char = [] # Value of index in list will be a character.

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
            self.W_all[l] = W_all.detach().clone().requires_grad_(True)
            self.U_all[l] = U_all.detach().clone().requires_grad_(True)
            self.B[l] = B.detach().clone().requires_grad_(True)
        
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
            h0 = torch.zeros(self.m[0], dtype=torch.float64) # shape (m, 1).

        X = torch.from_numpy(X)

        assert X.shape[0] == self.seq_len, f"X shape: {X.shape} != seq_len:{self.seq_len}"  # for catching errors.
        tau = self.seq_len

        apply_tanh = torch.nn.Tanh()
        apply_sigmoid = torch.nn.Sigmoid()
        apply_softmax = torch.nn.Softmax(dim=1)

        

        hprev = h0
        cprev = torch.zeros(self.m[0], dtype=torch.float64)

        
        Hs_L = torch.zeros(tau, self.m[self.L - 1], dtype=torch.float64) # the last Hs, which we use for s and P calculation.

        hprev = [torch.zeros(self.m[l], dtype=torch.float64) for l in range(self.L)]
        cprev = [torch.zeros(self.m[l], dtype=torch.float64) for l in range(self.L)]

        for t in range(tau):
            x_t = X[t]
            for l in range(self.L):
                #print(self.W_all[l].shape)
                #print(hprev[l].shape)
                a_t = self.W_all[l] @ hprev[l] + self.U_all[l] @ x_t + self.B[l]
                a_t = a_t.view(4, self.m[l])
                #print(a_t.shape)
                f_t = apply_sigmoid(a_t[0])
                i_t = apply_sigmoid(a_t[1])
                o_t = apply_sigmoid(a_t[2])
                c_hat_t = apply_tanh(a_t[3])

                c_t = f_t * cprev[l] + i_t * c_hat_t
                h_t = o_t * apply_tanh(c_t)

                hprev[l] = h_t
                cprev[l] = c_t

                x_t = h_t  # Feed to next layer

            # Save h_t of the final layer for output computation
            #print(hprev[-1].shape)
            #print(Hs_L.shape)
            Hs_L[t] = hprev[-1]

        S = torch.matmul(self.V, Hs_L.reshape(Hs_L.shape[1], Hs_L.shape[0])) + self.C.unsqueeze(1)
        P = apply_softmax(S.reshape(S.shape[1], S.shape[0]))
        # compute the loss
        loss = torch.mean(-torch.log(P[np.arange(tau), y]))  # use this line if storing inputs row-wise 
        return loss



    def backward(self, loss):
        # Zero gradients first
        for l in range(self.L):
            if self.W_all[l].grad is not None:
                self.W_all[l].grad.zero_()
            if self.U_all[l].grad is not None:
                self.U_all[l].grad.zero_()
            if self.B[l].grad is not None:
                self.B[l].grad.zero_()
        if self.V.grad is not None:
            self.V.grad.zero_()
        if self.C.grad is not None:
            self.C.grad.zero_()
        
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_([*self.W_all.values(), *self.U_all.values(), *self.B.values(), self.V, self.C], 5.0)



    def synth_text(self, x0, n, ind_to_char, char_to_ind, rng, h0 = None):
        '''
        Computes the forward pass of the LSTM model to make a prediction.
        :param x0: the first character
        :param n: length of the synthesized text
        :param rng: the previously initialized rng
        :return:
        '''
        #print(self.W_all[0][0])
        #print(self.U_all[0][0])
        #print(self.B[0])
        if h0 is None:
            h0 = torch.empty(self.m[0], 1, dtype=torch.float64) # shape (m, 1).

        apply_tanh = torch.nn.Tanh()
        apply_sigmoid = torch.nn.Sigmoid()
        apply_softmax = torch.nn.Softmax(dim=0)




        x = torch.zeros(self.K[0], dtype=torch.float64)
        x[char_to_ind[x0]] = 1

        synth_text = x0

        # initialize once, outside time loop
        h = [h0] + [torch.zeros(self.m[i], 1, dtype=torch.float64) for i in range(1, self.L)]
        c = [torch.zeros_like(h[i]) for i in range(self.L)]

        for _ in range(n):  # for each time step
            for l in range(self.L):
                a_t = self.W_all[l] @ h[l] + (self.U_all[l] @ x).unsqueeze(2) + self.B[l].unsqueeze(2)

                f_t = apply_sigmoid(a_t[0])
                i_t = apply_sigmoid(a_t[1])
                o_t = apply_sigmoid(a_t[2])
                c_hat = apply_tanh(a_t[3])

                c[l] = f_t * c[l] + i_t * c_hat
                h[l] = o_t * apply_tanh(c[l])

                x = h[l].squeeze()  # input for next layer

            s_t = self.V @ h[-1] + self.C.unsqueeze(1)
            p_t = apply_softmax(s_t)


            # sample (randomly from prob. dist. p_t), don't use in synthesized text until last layer.
            cp = np.cumsum(p_t.detach().numpy(), axis=0)
            a = rng.uniform(size=1)
            ii = np.argmax(cp - a > 0)

            x = torch.zeros(self.K[0], dtype=torch.float64)
            x[ii] = 1
            synth_text += ind_to_char[ii] # only synthesize in last layer.

        return synth_text


    def fit(self, epochs=5):
        '''
        Computes the gradient descent and trains the model using a number of epochs.

            NOTE: As of now, the X and Y one-hot encoded matrices are computed in Numpy,
            and the y-vector is computed using torch. Might want to convert the Numpy
            calculations to torch in the future to be consistent.
        :param epochs:
        :return:
        '''
        n_batches = len(self.X) // self.seq_len  # number of full batches
        trimmed_len = n_batches * self.seq_len
        X_trimmed = X[:trimmed_len]  # trim off the remainder, OBS: Losing some characters, maybe not necessary!
        Y_trimmed = X[1:trimmed_len + 1]
        y_trimmed = []
        for vec in self.X:
            ind = np.where(vec==1.0)[0] # adds the integer index of where the one-hot is 1.0.
            y_trimmed.append(int(ind.item()))
        y_trimmed = torch.tensor(y_trimmed[1:n_batches*self.seq_len+1])
        batches_X = X_trimmed.reshape(n_batches, self.seq_len, * X.shape[1:])
        batches_Y = Y_trimmed.reshape(n_batches, self.seq_len, * X.shape[1:])
        batches_y = y_trimmed.reshape(n_batches, self.seq_len)

        assert np.allclose(batches_Y[0, 0], batches_X[0, 1]), f"Data and labels are shifted wrong: Y: {batches_Y[0, 0]} != X: {batches_X[0, 1]}"
        
        smooth_loss = 0
        t = 1
        for _ in range(epochs):
            for i in tqdm(range(batches_X.shape[0])):
                loss = self.forward(batches_X[i], batches_y[i])
                self.backward(loss)

                smooth_loss = 0.999 * smooth_loss + 0.001 * loss.item() if t > 1 else loss.item()
                
                # Updates the weights
                with torch.no_grad():
                    for l in range(self.L):
                        self.W_all[l] -= self.W_all[l].grad * self.lr
                        self.U_all[l] -= self.U_all[l].grad * self.lr
                        self.B[l]     -= self.B[l].grad * self.lr
                    self.V -= self.V.grad * self.lr
                    self.C -= self.C.grad * self.lr
                if i % 1000 == 0:
                     print("\n----------------------")
                     print("W Gradient magnitudes: " + str(self.W_all[l].grad.norm().item()))
                     print("LOSS: " + str(smooth_loss))
                     print(self.synth_text("a", 25, ind2char, char2ind, rng))
                     print("----------------------\n")
                
                t += 1 # used for smooth loss
           



rng = np.random.default_rng()
# get the BitGenerator used by default
BitGen = type(rng.bit_generator)
# use the state from a fresh bit generator
seed = 1
rng.bit_generator.state = BitGen(seed).state

data, unique_chars = ReadData()
char2ind, ind2char = GetDicts(unique_chars)
X = ConvertToOneHot(data, char2ind)

X_seq = X[0:25]
y_seq = data[1:26] # not one-hot encoded.
y_seq_indices = [char2ind[char] for char in y_seq]

lstm = LSTM(X[0: 26], m=[100, 50], n_layers=2)


#print(lstm.W_all)
"""
loss = lstm.forward(X_seq, y_seq_indices)
grads = lstm.backward(loss)
print(lstm.C.grad)
print(lstm.W_all[1].grad[0])
print(torch.any(lstm.U_all[0].grad != 0))
"""#synth_text = lstm.synth_text("a", 25, ind2char, char2ind, rng)
#print(synth_text)
lstm.fit()


