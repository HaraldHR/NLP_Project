import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RNN(nn.Module):


    def __init__(self, input_size, hidden_size, output_size, num_layers=1, nonlinearity='tanh'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,  # 'tanh' or 'relu'
            batch_first=False  # input shape (seq_len, batch, input_size)
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden=None):

        rnn_out, hidden = self.rnn(input_seq, hidden)
        # Apply linear layer to each time step
        output = self.fc(rnn_out)  # (seq_len, batch, output_size)
        return output, hidden

    def train_model(self, X_input, Y_input, num_epochs, seq_len=50, learning_rate=0.001):

        total_length = X_input.size(0)
        input_size = X_input.size(1)

        # Ensure X and Y have the same length
        assert total_length == Y_input.size(0), "Input and target must have the same length"

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)


        num_chunks = (total_length - 1) // seq_len

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for c in range(num_chunks):
                start = c * seq_len
                end = start + seq_len + 1

                input_seq = X_input[start:end].unsqueeze(1)  # (seq_len, 1, input_size)
                target_seq = Y_input[start:end].unsqueeze(1)  # (seq_len, 1, input_size)

                optimizer.zero_grad()
                output_seq, hidden = self.forward(input_seq)
                loss = criterion(output_seq, target_seq)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"Chunk [{c + 1}/{num_chunks}], Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / num_chunks
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

