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
from DataProcessing import ReadData, GetDicts, ConvertToOneHot, TrainValSplit, GetBatches, TrainTestSplit

def grid_search_rnn(X_input, unique_chars, learning_rates, seq_lengths, batch_sizes, num_epochs=10, hidden_dim=128):
    results = []
    best_val_loss = float('inf')
    best_model_state = None
    best_lr = None
    best_dim = None
    best_batch_size = None

    X_train, X_val = TrainValSplit(X_input, 0.2)

    total_iteration = len(learning_rates) * len(seq_lengths) * len(batch_sizes)
    i = 0
    for lr in tqdm(learning_rates, desc="LR loop"):
        for seq in tqdm(seq_lengths, desc="Hidden dims loop"):
            for batch_size in tqdm(batch_sizes, desc="Batch size loop"):

                X_train_batches, Y_train_batches = GetBatches(X_train.clone(), seq, batch_size)
                X_val_batches, Y_val_batches = GetBatches(X_val.clone(), seq, batch_size)

                model = RNN(
                    input_size=X_input.shape[1],
                    hidden_size=hidden_dim,
                    output_size=X_input.shape[1],
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

    print(f"Best combo: LR={best_lr}, seq={best_dim}, Batch={best_batch_size}, Loss={best_val_loss:.4f}")
    with open("grid_search_rnn.txt", "a") as f:
        f.write(f"Search at time: {datetime.datetime.now()}")
        f.write(f"\n------- Epochs: {num_epochs} -------\n")
        for result in results:
            f.write(f"Sequence Length: {result[0]}, Learning rate: {result[1]}, Batch size: {result[2]}, Validation Loss: {result[3]}\n")

    return best_lr, best_dim, best_model_state, results


def preprocess_data():
    # Get the raw data and unique characters
    data, unique_chars = ReadData("shakespeare.txt")

    # Create a mapping of character to index
    char2ind = {char: idx for idx, char in enumerate(unique_chars)}
    ind2char = {idx: char for idx, char in enumerate(unique_chars)}

    # Convert data to a sequence of indices
    data_indices = [char2ind[char] for char in data]

    # Convert indices to tensor
    X_input = torch.tensor(data_indices, dtype=torch.long)

    # One-hot encode the data
    X_one_hot = F.one_hot(X_input, num_classes=len(unique_chars)).float()

    return X_one_hot, unique_chars

X_data, unique_chars = preprocess_data()


char_to_ind, ind_to_char = GetDicts(unique_chars)


X_train, X_test = TrainTestSplit(X_data, train_size=0.8)
X_train, X_val = TrainValSplit(X_data, val_size=0.2)

X_train_batches, Y_train_batches = GetBatches(X_train, seq_len=50, batch_size=64)

X_val_batches, Y_val_batches = GetBatches(X_val, seq_len=50, batch_size=64)

learning_rates = [ 1e-4]
seq_lengths = [50]
batch_sizes = [32]
grid_search_rnn(X_train, unique_chars, learning_rates, seq_lengths, batch_sizes, num_epochs=5)

