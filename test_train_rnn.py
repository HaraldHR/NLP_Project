

from DataProcessing import ReadData, GetDicts, ConvertToOneHot



import torch

import numpy as np

from model_RNN import RNN

from synthesizer import synthesize_text


# Load the data
data, unique_chars = ReadData()
char_to_ind, ind_to_char = GetDicts(unique_chars)
X = ConvertToOneHot(data, char_to_ind)



print(X.shape)

# Convert to torch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)


#Y_tensor = np.roll(X, shift=-1, axis=0)
Y_tensor = torch.tensor(X_tensor, dtype=torch.float32)

# Hyperparameters
input_size = len(char_to_ind)  # Number of unique characters
hidden_size = 128  # Hidden layer size
output_size = len(char_to_ind)  # Output size is same as number of unique characters
num_layers = 2  # Number of RNN layers
num_epochs = 3  # Number of epochs for training

# Create the model
model = RNN(input_size, hidden_size, output_size, num_layers)

subset_length = 200000  # or any length you want

# Slice the subset
X_subset = X_tensor[:subset_length]
Y_subset = Y_tensor[:subset_length]

# Train the model
best_loss, best_model_state_dict = model.train_model(X_subset, X_subset, num_epochs=num_epochs, seq_len=50, learning_rate=1e-4)


#checkpoint = torch.load("best_rnn_model.pth")
#model_dict = model.state_dict()
#print("FIRST")
#print(model_dict)

# Filter out unnecessary keys from the checkpoint
#filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}

#print(filtered_checkpoint)

# Load the filtered state dict into your model
#model_dict.update(filtered_checkpoint)
#print("SECODN")
#print(model_dict)
model.load_state_dict(best_model_state_dict)
print(best_model_state_dict)
#print(model.state_dict())



text = synthesize_text(model, "A", char_to_ind, ind_to_char, 200)

print(text)

print(f"Best Loss:{best_loss}")


"""

from DataProcessing import ReadData, GetDicts
from model_RNN import ManualRNN
import torch

# Load the data
data, unique_chars = ReadData()
char_to_ind, ind_to_char = GetDicts(unique_chars)

# Hyperparameters
hidden_size = 128
num_epochs = 5
seq_length = 25
learning_rate = 0.001

# Limit training data for faster iteration (you can increase this later)
book_data = data[:100000]

# Initialize the model
rnn = ManualRNN(input_size=len(char_to_ind), hidden_size=hidden_size, output_size=len(char_to_ind))

# Train the model
rnn, best_params, best_loss, loss_history = rnn.train_rnn(
    rnn, book_data, char_to_ind, ind_to_char,
    num_epochs=num_epochs,
    seq_length=seq_length,
    eta=learning_rate
)

# Restore best weights
rnn.params = best_params

# Generate text
start_char = "A"
x0 = rnn.one_hot_encode(char_to_ind[start_char], len(char_to_ind))
h0 = torch.zeros(hidden_size, 1)
sample = rnn.synthesize_seq(rnn, h0, x0, length=200, ind_to_char=ind_to_char)

print("\n--- Final Sample ---")
print(sample)

"""
