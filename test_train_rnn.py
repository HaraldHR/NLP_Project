

from DataProcessing import ReadData, GetDicts, ConvertToOneHot



import torch

import numpy as np

from model_RNN import RNN

from synthesizer import synthesize_text


# Load the data
data, unique_chars = ReadData("goblet_book.txt")
char_to_ind, ind_to_char = GetDicts(unique_chars)
X = ConvertToOneHot(data, char_to_ind)



print(X.shape)

# Convert to torch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)


#Y_tensor = np.roll(X, shift=-1, axis=0)
#Y_tensor = torch.tensor(X_tensor, dtype=torch.float32)

# Hyperparameters
input_size = len(char_to_ind)
hidden_size = 128
output_size = len(char_to_ind)
num_layers = 3
num_epochs = 4

model = RNN(input_size, hidden_size, output_size, num_layers)

subset_length = 100000

# Slice the subset
X_subset = X_tensor[:subset_length]
#Y_subset = Y_tensor[:subset_length]

# Train the model
best_loss, best_model_state_dict = model.train_model(X_subset, X_subset, num_epochs=num_epochs, seq_len=50, learning_rate=0.001)


model.load_state_dict(best_model_state_dict)
print(best_model_state_dict)


initial_char = 'H'
x0 = np.zeros(len(char_to_ind), dtype=np.float32)
x0[char_to_ind[initial_char]] = 1.0

text = synthesize_text(model, x0, 10000, ind_to_char, char_to_ind)

print(text)

print(f"Best Loss:{best_loss}")

