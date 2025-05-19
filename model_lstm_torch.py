import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from DataProcessing import *  # Assuming ReadData is in DataProcessing.py
import LSTM_search
import copy
from tqdm import tqdm
from DataVisualization import LossPlotter

import os

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, unique_chars, batch_size, seq_len,  num_layers=1, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.char2ind = {char: idx for idx, char in enumerate(unique_chars)}
        self.ind2char = {idx: char for idx, char in enumerate(unique_chars)}
        self.batch_size = batch_size
        self.seq_len = seq_len
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,  # input shape (seq_len, batch, input_size)
            dropout=dropout     # Dropout between layers (only for num_layers > 1)
        )

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden=None):
        # Pass the input through the LSTM layer
        lstm_out, hidden = self.lstm(input_seq, hidden)
        # Apply the fully connected layer to the output of the LSTM
        output = self.fc(lstm_out)  # (seq_len, batch, output_size)
        
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state (h0, c0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h0, c0)
    
    def nucleus_sampling(self, probs, p=0.9):
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # Sum up probs
        cumulative_probs = torch.cumsum(sorted_probs, dim=1)

        # Find cutoff where cumulative probability exceeds p

        cutoff = 0
        for sum in cumulative_probs.squeeze():
            cutoff += 1
            if sum.item() > p:
                break
        #print(sorted_probs.shape)
        #print(sorted_indices.shape)
        top_probs = sorted_probs[0, :cutoff + 1]
        top_indices = sorted_indices[0, :cutoff + 1]
        # Renormalize the probabilities
        top_probs = top_probs / top_probs.sum()
        # Sample from top-p
        selected_idx = torch.multinomial(top_probs, 1)
        return top_indices.squeeze()[selected_idx].item()
    
    def temperature_sampling(self, output, temp=0.8):
        probs = F.softmax(output / temp, dim=0)
        next_char_id = torch.multinomial(probs, 1).item() # samples from distribution.
        return next_char_id

    def synth_text(self, start_char, char2ind, ind2char, seq_len=100):
        self.eval()  # Set the model to evaluation mode

        # Convert start_char to one-hot encoding
        start_idx = char2ind[start_char]
        input_char = torch.tensor([start_idx], dtype=torch.long).unsqueeze(0)  # shape (1, 1)

        # One-hot encode the character
        input_one_hot = F.one_hot(input_char, num_classes=len(char2ind)).float()

        # Initialize hidden state
        hidden = self.init_hidden(batch_size=1)

        # Start generating text
        generated_text = [start_char]
        
        for _ in range(seq_len):
            # Get the output from the LSTM forward pass
            output, hidden = self(input_one_hot, hidden)
            
            # Apply softmax to get probabilities for the next character

            output_probs = F.softmax(output[-1], dim=-1)  # Use the last output for prediction
            next_char_idx = torch.multinomial(output_probs, 1).item() # Pure Sampling
            #next_char_idx = self.nucleus_sampling(output_probs) # Nucleus Sampling
            #next_char_idx = self.temperature_sampling(output.squeeze())

            # Get the next character
            next_char = ind2char[next_char_idx]
            generated_text.append(next_char)

            # Update the input for the next time step: use the predicted character
            input_char = torch.tensor([next_char_idx], dtype=torch.long).unsqueeze(0)
            input_one_hot = F.one_hot(input_char, num_classes=len(char2ind)).float()

        self.train()

        return ''.join(generated_text)



    def train_model(self, X_train_batches, Y_train_batches, X_val_batches, Y_val_batches, num_epochs, learning_rate=0.001, best_loss_ever = 10000):
        output_size = self.fc.out_features

        
        best_model_state_dict = {}
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        n = 0

        loss_train = []
        loss_val = []
        epochs = []

        best_loss = best_loss_ever
        best_model_state_dict = None

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            hidden = None

            epoch_loss = 0
            for i in range(X_train_batches.shape[0]):
                
                #print(f"Batch: {i+1}")
                X_batch = X_train_batches[i] # (seq_len, 1, input_size)
                Y_batch = Y_train_batches[i]
                
                target_seq = torch.argmax(Y_batch, dim=2)

                optimizer.zero_grad()

                # Initialize the hidden and cell states if None
                if hidden is None:
                    hidden = self.init_hidden(batch_size=self.batch_size)

                output_seq, hidden = self.forward(X_batch, hidden)
                # Detach hidden states to avoid backprop through the entire training history
                if hidden is not None:
                    hidden = tuple([h.detach() for h in hidden])

                # Reshape output: (seq_len, batch, output_size) → (seq_len * batch, output_size)
                output_seq = output_seq.view(-1, output_size)
                target_seq = target_seq.view(-1)


                loss = criterion(output_seq, target_seq) # cross entropy loss
                epoch_loss += loss.item() # for average
                loss.backward()

                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)

                optimizer.step()

                if n == 0:
                    smooth_loss_train = loss.item()
                else:
                    smooth_loss_train = 0.999 * smooth_loss_train + 0.001 * loss.item()

                """
                if n % 500 == 0:
                    #print(f"Iteration [{n + 1}/{n}], Loss: {smooth_loss_train:.4f}")
                    print(self.synth_text("A", self.char2ind, self.ind2char, seq_len=100))
                """
                n += 1
            
            # check on validation set at the end of each epoch:
            self.eval()
            val_loss_sum = 0
            for j in range(X_val_batches.shape[0]):
                val_output, _ = self.forward(X_val_batches[j], None)
                val_targets = torch.argmax(Y_val_batches[j], dim = 2)
                val_output = val_output.view(-1, output_size)
                val_targets = val_targets.view(-1)
                val_loss = criterion(val_output, val_targets) # cross entropy loss
                val_loss_sum += val_loss.item()
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_model_state_dict = self.state_dict()
            self.train()
            train_loss_avg = epoch_loss / X_train_batches.shape[0]
            val_loss_avg = val_loss_sum / X_val_batches.shape[0]
            

            loss_train.append(train_loss_avg)
            loss_val.append(val_loss_avg)
            epochs.append(epoch)

                    
            #steps_for_plotting = torch.arange(0, n, 500)
            #avg_loss = epoch_loss / num_chunks
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss_avg}")
            #print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
        print(f"Lowest avg loss found in epoch {np.argmin(loss_val)}")
        return best_loss, best_model_state_dict, epochs, loss_train, loss_val


def preprocess_data():
    # Get the raw data and unique characters
    data, unique_chars = ReadData()
    
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


def main():
    # Parameters
    input_size = 65  # Modify based on unique_chars length
    hidden_size = 256
    output_size = 65  # Modify based on unique_chars length
    num_layers = 2
    seq_len = 50
    num_epochs = 40
    learning_rate = 0.001
    batch_size = 32

    best_model_path = "best_lstm_model.pth"
    best_loss_path = "best_loss_lstm.txt"

    # Load previous best loss if file exists
    if os.path.exists("best_loss_lstm.txt"):
        with open("best_loss_lstm.txt", 'r') as f:
            best_loss_ever = float(f.read().strip())
    else:
        best_loss_ever = float('inf')  # If not found, use infinity as the initial best
    
    # Preprocess data
    X_data, unique_chars = preprocess_data()

    X_train, X_val, X_test = TrainValTestSplit(X_data)

    X_train_batches, Y_train_batches = GetBatches(X_train, seq_len=seq_len, batch_size=batch_size)

    X_val_batches, Y_val_batches = GetBatches(X_val, seq_len=seq_len, batch_size=batch_size) # Simply for input shape for the forwardpass, we make on big batch

    """
    learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3]
    seq_lengths = [25, 50, 75, 100]
    batch_sizes = [16, 32, 64, 128]
    LSTM_search.grid_search_lstm(X_train, unique_chars, learning_rates, seq_lengths, batch_sizes, num_epochs=10)
    """
    
    # Initialize the LSTM model
    model = LSTM(input_size=input_size, hidden_size=hidden_size, unique_chars = unique_chars, output_size=output_size, num_layers=num_layers, batch_size = batch_size, seq_len = seq_len)
    
    # Train the model
    best_loss_after, best_model_state_dict, epochs_for_plot, train_loss, val_loss = model.train_model(X_train_batches, Y_train_batches, X_val_batches, Y_val_batches, num_epochs, learning_rate=learning_rate, best_loss_ever = best_loss_ever)
    if best_loss_after < best_loss_ever:
        # Write best result to to files
        torch.save(best_model_state_dict, best_model_path)
        with open(best_loss_path, "w") as f:
            f.write(str(best_loss_after))
        print(f"Best loss after training: {best_loss_after:.4f}")

    LossPlotter.plot_losses(train_loss, val_loss, epochs_for_plot)

if __name__ == '__main__':
    main()
