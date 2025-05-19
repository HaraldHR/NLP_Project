from collections import defaultdict

import numpy as np
import torch
from word_embeddings.BPE import BPE


def ReadData(file):
    if file is None:
        fid = open('shakespeare.txt', "r")
        book_data = fid.read()
        unique_chars = list(set(book_data))
        return book_data, unique_chars
    else:
        with open(file, 'r') as f:
            for row in f:
                print(row)


def create_word_vocab(file):
    vocab = []
    with open(file, 'r') as file:
        for row in file:
            row = row.strip()
            word = row.split(' ')[0]
            if word not in vocab:
                vocab.append(word)
    return vocab


def tokenize_data(book_data, vocab_size=500):
    bpe = BPE()
    bpe.set_vocab_length(vocab_size)
    print(f"Tokenizing data with vocab size of {vocab_size}")
    vocab, tokens = bpe.tokenize(book_data)
    print("Tokenization finished.")
    return vocab, tokens


def get_ids(tokens, word2id):
    ids = []
    for token in tokens:
        ids.append(int(word2id[token]))
    return ids


def get_embeddings(vector_file):
    '''
    Reads a file with vector embeddings. Creates a dictionairy with chars as keys,
    and embedding vectors as values.
    :param ids:
    :param id2word: dictionairy that maps an ID to a word in characters.
    :return: word2id dict, id2word dict, ID to embedding dict
    '''
    word2id = defaultdict(int)
    id2word = {}
    id2embed = {}

    with open(vector_file, 'r') as file:
        for row in file:
            row = row.strip()
            row = row.split(' ')
            new_id = int(len(word2id.keys()))
            word2id[row[0]] = new_id # adds a new ID.
            id2word[new_id] = row[0]
            embedding = np.array([float(x) for x in row[1:]]) # can convert to np or torch float64 (to create long)
            id2embed[new_id] = embedding
    return word2id, id2word, id2embed



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
    X_test = X[split_index:]
    return X_train, X_test



def TrainValSplit(X, val_size): # Simply for clearer naming
    X_train, X_val = TrainTestSplit(X, 1-val_size)
    return X_train, X_val



def get_embed_tensor(id2embed):
    embed_dim = id2embed[0].shape[0]
    print(f"Embed dim: {embed_dim}")
    emb_mat = torch.empty(len(id2embed.keys()), embed_dim)
    for id in id2embed.keys():
        emb_mat[id] = torch.tensor(id2embed[id])
    return emb_mat



def GetBatches(X, seq_len, batch_size, use_embeddings=False):
    total_len = len(X)
    num_full_batches = (total_len - 1) // (batch_size * seq_len)

    trimmed_len = num_full_batches * batch_size * seq_len

    # If using embeddings, X will be a list of IDs.
    # If NOT using embeddings, X will be a list of one-hot-encodings.
    X_input = torch.from_numpy(np.array(X[:trimmed_len]))
    Y_input = torch.from_numpy(np.array(X[1:trimmed_len + 1]))
    #print(X_input.shape)

    if use_embeddings:
        print("Using embeddings")
        X_batches = torch.empty(num_full_batches, batch_size, seq_len, dtype=torch.long)
        Y_batches = torch.empty(num_full_batches, batch_size, seq_len, dtype=torch.long)
        e = 0
        for i in range(num_full_batches):
            for j in range(batch_size):
                X_batches[i, j] = X_input[e : e + seq_len]
                Y_batches[i, j] = Y_input[e : e + seq_len]
                e += seq_len
    else:
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

file = "./word_embeddings/vectors_bpe_500.txt"
word2id, id2word, id2embedding = get_embeddings(file)
book_data, _ = ReadData()
vocab, tokens = tokenize_data(book_data, vocab_size=100)
ids = get_ids(tokens, word2id)
X, Y = GetBatches(ids, 25, 32, use_embeddings=True)
print(X[1].shape)
print(Y[1].dtype)

emb_mat = get_embed_tensor(id2embedding)
