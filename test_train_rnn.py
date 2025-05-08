from DataProcessing import ReadData, GetDicts, ConvertToOneHot

import torch

import numpy as np

from model_RNN import RNN


# Load the data
data, unique_chars = ReadData()
char_to_ind, ind_to_char = GetDicts(unique_chars)
X = ConvertToOneHot(data, char_to_ind)



print(X.shape)

# Convert to torch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)


Y_tensor = np.roll(X, shift=-1, axis=0)
Y_tensor = torch.tensor(Y_tensor, dtype=torch.float32)

# Hyperparameters
input_size = len(char_to_ind)  # Number of unique characters
hidden_size = 128  # Hidden layer size
output_size = len(char_to_ind)  # Output size is same as number of unique characters
num_layers = 1  # Number of RNN layers
num_epochs = 1  # Number of epochs for training

# Create the model
model = RNN(input_size, hidden_size, output_size, num_layers)

subset_length = 10000  # or any length you want

# Slice the subset
X_subset = X_tensor[:subset_length]
Y_subset = Y_tensor[:subset_length]

# Train the model
model.train_model(X_subset, Y_subset, num_epochs=num_epochs, seq_len=50, learning_rate=0.001)