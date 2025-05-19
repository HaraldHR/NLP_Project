


from DataProcessing import ReadData, GetDicts, ConvertToOneHot, TrainValTestSplit, TrainValSplit, GetBatches
import torch.nn.functional as F
from DataVisualization import LossPlotter
from model_evaluation import *


import pickle
import torch

import numpy as np

from model_RNN import RNN

from synthesizer import synthesize_text


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

with open("dicts.pkl", "rb") as f:
    vocab = pickle.load(f)
    char2ind = vocab['char2ind']
    ind2char = vocab['ind2char']


X_train, X_val, X_test = TrainValTestSplit(X_data, 0.8, 0.1)

X_train_batches, Y_train_batches = GetBatches(X_train, seq_len=50, batch_size=32)

X_val_batches, Y_val_batches = GetBatches(X_val, seq_len=50, batch_size=32)
data_str, _ = ReadData("shakespeare.txt")
processor = TextProcessor(data_str)



model = RNN(
    input_size=X_data.shape[1],
    hidden_size=256,
    output_size=X_data.shape[1],
    num_layers=2,
    nonlinearity='tanh',
)

loss, loss_val, epochs, best_loss, model_state = model.train_model(
    X_train_batches,
    Y_train_batches,
    X_val_batches,
    Y_val_batches,
    num_epochs=3,
    learning_rate=0.001,
    best_loss_ever=10000  # Doesn't matter in search
)


torch.save(model_state, "best_rnn_model.pth")
start_char = 'H'
x0 = F.one_hot(torch.tensor([char2ind[start_char]]), num_classes=len(char2ind)).float()[0]
generated = model.synthesize_text(x0, n=1000, ind_to_char=ind2char, char_to_ind=char2ind)
run_text_quality_tests(generated, processor, "No Model, Just Test")


print("Generated Text:")
print(generated)
print(best_loss)
with open("generated_text_rnn.txt", "w", encoding="utf-8") as f:
    f.write("Generated Text:\n")
    f.write(generated + "\n\n")
    f.write(f"Best Loss: {best_loss:.4f}\n")

LossPlotter.plot_losses(loss, loss_val, epochs)