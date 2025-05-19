import numpy as np
import torch


def ReadData():
    fid = open('shakespeare.txt', "r")
    book_data = fid.read()
    fid.close()
    unique_chars = list(set(book_data))
    return book_data, unique_chars

def GetDicts(unique_chars):
    char_to_ind = {char: i for i, char in enumerate(unique_chars)}
    ind_to_char = {i: char for i, char in enumerate(unique_chars)}
    return char_to_ind, ind_to_char


def ConvertToOneHot(chars, char_to_ind):
    K = len(char_to_ind)
    one_hot = np.zeros((len(chars), K))
    for i in range(len(chars)): # Make into one-hot encoding representation.
        one_hot[i, char_to_ind[chars[i]]] = 1 
    return one_hot

def TrainValTestSplit(X, train_size=0.8, val_size=0.1): # X is one-hot encoded, train_size is a fraction.
    n = X.shape[0]
    split_index_train = int(n * train_size)
    split_index_val = split_index_train + int(n * val_size)
    X_train = X[:split_index_train]
    X_val = X[split_index_train:split_index_val]
    X_test = X[split_index_val:]
    return X_train, X_val, X_test

def TrainValSplit(X, val_size): # Simply for clearer naming
    X_train, X_val = TrainTestSplit(X, 1-val_size)
    return X_train, X_val

def GetBatches(X, seq_len, batch_size):
    total_len = len(X)
    num_full_batches = (total_len - 1) // (batch_size * seq_len)

    trimmed_len = num_full_batches * batch_size * seq_len
    X_input = torch.from_numpy(np.array(X[:trimmed_len]))
    Y_input = torch.from_numpy(np.array(X[1:trimmed_len + 1]))
    #print(X_input.shape)
    X_batches = torch.empty(num_full_batches, seq_len, batch_size, X.shape[1])
    Y_batches = torch.empty(num_full_batches, seq_len, batch_size, X.shape[1])
    #Y_batches
    # reshape into batch_size rows
    batches = []
    e = 0 # current seq index start
    for i in range(num_full_batches):
        for j in range(batch_size): # batch nr i
            #print(X_batches[:, i].shape)
            #print(X_input[e : e + seq_len].shape)
            X_batches[i, :, j] = X_input[e : e + seq_len]
            Y_batches[i, :, j] = Y_input[e : e + seq_len]
            e += seq_len
    
    return X_batches, Y_batches # a tuple for each batch

   
"""
data, unique_chars = ReadData()
char_to_ind, ind_to_char = GetDicts(unique_chars)
X = ConvertToOneHot(data, char_to_ind)

batches = GetTrainBatches(X, 25)
print(batches[0][0])
"""
