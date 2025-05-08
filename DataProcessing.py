import numpy as np

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


data, unique_chars = ReadData()
char_to_ind, ind_to_char = GetDicts(unique_chars)
X = ConvertToOneHot(data, char_to_ind)
print(X.shape)
