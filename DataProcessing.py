import numpy as np

def ReadData(filepath):
    fid = open(filepath, "r")
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

def TrainTestSplit(X, train_size): # X is one-hot encoded, train_size is a fraction.
    n = X.shape[0]
    split_index = int(n * train_size)
    X_train = X[:split_index]
    X_test = X[split_index]
    return X_train, X_test

def TrainValSplit(X, val_size): # Simply for clearer naming
    return TrainTestSplit(X, val_size)

def GetTrainBatches(X, batch_size):
    n = len(X)
    n_batches = n // batch_size  # number of full batches
    trimmed_len = n_batches * batch_size

    X_trimmed = X[:trimmed_len]  # trim off the remainder, OBS: Losing some characters, maybe not necessary!
    batches = X_trimmed.reshape(n_batches, batch_size, *X.shape[1:])

    return batches # maybe shuffle batches later.

   
"""
data, unique_chars = ReadData()
char_to_ind, ind_to_char = GetDicts(unique_chars)
X = ConvertToOneHot(data, char_to_ind)

batches = GetTrainBatches(X, 25)
print(batches[0][0])
"""
