import re
from model_RNN import RNN
import torch
import pickle
import torch.nn.functional as F
from DataProcessing import ReadData, TrainValTestSplit, GetBatches

def find_lowest_loss_line(file_path):
    lowest_loss = float('inf')
    best_line = ""

    pattern = re.compile(r"Validation Loss: ([0-9.]+)")

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                loss = float(match.group(1))
                if loss < lowest_loss:
                    lowest_loss = loss
                    best_line = line.strip()

    if best_line:
        print(f"Lowest Validation Loss: {lowest_loss}")
        print(f"Line: {best_line}")
    else:
        print("No 'Validation Loss' lines found.")

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


def testFinalModel():
    X_data, unique_chars = preprocess_data()

    with open("dicts.pkl", "rb") as f:
        vocab = pickle.load(f)
        char2ind = vocab['char2ind']
        ind2char = vocab['ind2char']

    input_size = output_size = X_data.shape[1]
    hidden_size = 256
    num_layers = 2
    batch_size = 32



    model = RNN(input_size=input_size, hidden_size=hidden_size,
                    output_size=output_size, num_layers=num_layers)
    model.load_state_dict(torch.load("best_rnn_model.pth"))

    X_train, X_val, X_test = TrainValTestSplit(X_data, 0.8, 0.1)
    X_train = torch.cat((X_train, X_val))

    X_train_batches, Y_train_batches = GetBatches(X_train, seq_len=100, batch_size=16)

    X_tes_batches, Y_test_batches = GetBatches(X_test, seq_len=100, batch_size=16)

    print(model.forward_loss(X_tes_batches, Y_test_batches))

# Example usage:
#find_lowest_loss_line("grid_search_rnn.txt")

testFinalModel()

