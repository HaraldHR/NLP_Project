


from DataProcessing import ReadData, GetDicts, ConvertToOneHot, TrainTestSplit, TrainValSplit, GetBatches
import torch.nn.functional as F
from DataVisualization import LossPlotter



import torch

import numpy as np

from model_RNN import RNN

from synthesizer import synthesize_text


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

X_train_batches, Y_train_batches = GetBatches(X_train, seq_len=50, batch_size=32)

X_val_batches, Y_val_batches = GetBatches(X_val, seq_len=50, batch_size=32)



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
    num_epochs=10,
    learning_rate=0.001,
    best_loss_ever=10000  # Doesn't matter in search
)


torch.save(model_state, "best_rnn_model.pth")
start_char = 'H'
x0 = F.one_hot(torch.tensor([char_to_ind[start_char]]), num_classes=len(char_to_ind)).float()[0]
generated = model.synthesize_text(x0, n=1000, ind_to_char=ind_to_char, char_to_ind=char_to_ind)

print("Generated Text:")
print(generated)
print(best_loss)
with open("generated_text_rnn.txt", "w", encoding="utf-8") as f:
    f.write("Generated Text:\n")
    f.write(generated + "\n\n")
    f.write(f"Best Loss: {best_loss:.4f}\n")

LossPlotter.plot_losses(loss, loss_val, epochs)