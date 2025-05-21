import torch
import itertools
import copy
import torch
import io
import contextlib
import datetime
from tqdm import tqdm
import torch.nn.functional as F
from model_RNN import RNN
import pickle
from DataProcessing import ReadData, GetDicts, ConvertToOneHot, TrainValSplit, GetBatches, TrainValTestSplit

def grid_search_rnn(X_train, X_val, unique_chars, learning_rates, seq_lengths, batch_sizes, num_epochs=10, hidden_dim=128):
    results = []
    best_val_loss = float('inf')
    best_model_state = None
    best_lr = None
    best_dim = None
    best_batch_size = None

    total_iteration = len(learning_rates) * len(seq_lengths) * len(batch_sizes)
    i = 0
    with open("grid_search_rnn.txt", "a") as f:
        f.write(f"Search at time: {datetime.datetime.now()}")
        f.write(f"\n------- Epochs: {num_epochs} -------\n")

    for lr in tqdm(learning_rates, desc="LR loop"):
        for seq in tqdm(seq_lengths, desc="Hidden dims loop"):
            for batch_size in tqdm(batch_sizes, desc="Batch size loop"):

                X_train_batches, Y_train_batches = GetBatches(X_train.clone(), seq, batch_size)
                X_val_batches, Y_val_batches = GetBatches(X_val.clone(), seq, batch_size)

                model = RNN(
                    input_size=X_train.shape[1],
                    hidden_size=hidden_dim,
                    output_size=X_train.shape[1],
                    num_layers=2,
                    nonlinearity='tanh',
                )
                model.batch_size = batch_size  # Add if your model uses this

                with contextlib.redirect_stdout(io.StringIO()):  # Suppress training prints
                    loss, loss_val, epochs, best_loss, model_state = model.train_model(
                        X_train_batches,
                        Y_train_batches,
                        X_val_batches,
                        Y_val_batches,
                        num_epochs=num_epochs,
                        learning_rate=lr,
                        best_loss_ever=10000  # Doesn't matter in search
                    )

                min_val_loss = min(loss_val)
                results.append((seq, lr, batch_size, min_val_loss))

                if float(min_val_loss) < best_val_loss:
                    best_val_loss = min_val_loss
                    best_lr = lr
                    best_dim = seq
                    best_batch_size = batch_size
                    best_model_state = model_state
                i += 1
                with open("grid_search_rnn.txt", "a") as f:
                    f.write(
                        f"Sequence Length: {seq}, Learning rate: {lr}, Batch size: {batch_size}, Validation Loss: {min_val_loss}\n")

    print(f"Best combo: LR={best_lr}, seq={best_dim}, Batch={best_batch_size}, Loss={best_val_loss:.4f}")

    return best_lr, best_dim, best_model_state, results


def preprocess_data():
    # Get the raw data and unique characters
    data, unique_chars = ReadData("shakespeare.txt")

    # Create a mapping of character to index
    with open("dicts.pkl", "rb") as f:
        vocab = pickle.load(f)
        char2ind = vocab['char2ind']
        ind2char = vocab['ind2char']

    # Convert data to a sequence of indices
    data_indices = [char2ind[char] for char in data]

    # Convert indices to tensor
    X_input = torch.tensor(data_indices, dtype=torch.long)

    # One-hot encode the data
    X_one_hot = F.one_hot(X_input, num_classes=len(unique_chars)).float()

    return X_one_hot, unique_chars

X_data, unique_chars = preprocess_data()


char_to_ind, ind_to_char = GetDicts(unique_chars)

X_train, X_val, X_test = TrainValTestSplit(X_data, 0.8, 0.1)


X_train_batches, Y_train_batches = GetBatches(X_train, seq_len=50, batch_size=64)

X_val_batches, Y_val_batches = GetBatches(X_val, seq_len=50, batch_size=64)

learning_rates = [1e-3, 3e-3, 8e-4, 6e-4, 3e-4 ,1e-4, 8e-5, 3e-5, 1e-5]
seq_lengths = [25, 50, 75, 100]
batch_sizes = [16, 32, 64, 128 ]
grid_search_rnn(X_train, X_val, unique_chars, learning_rates, seq_lengths, batch_sizes, num_epochs=10)

