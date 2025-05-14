import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F


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
        output_size = self.fc.out_features

        Y_indices = torch.argmax(Y_input, dim=-1)

        best_loss = float('inf')  # Initialize with a high value
        best_model_path = "best_rnn_model.pth"
        best_model_state_dict = {}

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        num_chunks = (total_length - 1) // seq_len
        n = 0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            hidden = None

            for c in range(num_chunks):
                start = c * seq_len
                end = start + seq_len

                input_seq = X_input[start:end].unsqueeze(1)  # (seq_len, 1, input_size)
                target_seq = Y_indices[start + 1:end + 1]  # (seq_len,)

                optimizer.zero_grad()
                output_seq, hidden = self.forward(input_seq, hidden)
                if hidden is not None:
                    hidden = hidden.detach()

                # Reshape: (seq_len, batch, output_size) → (seq_len * batch, output_size)
                output_seq = output_seq.view(-1, output_size)
                target_seq = target_seq.view(-1)

                loss = criterion(output_seq, target_seq)
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)

                optimizer.step()
                if n == 0:
                    smooth_loss = loss.item()
                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                epoch_loss += smooth_loss

                if smooth_loss < best_loss:
                    best_loss = smooth_loss
                    #torch.save(self.state_dict(), best_model_path)
                    best_model_state_dict = self.state_dict()
                    print(f"Best model saved with loss: {best_loss:.4f}")
                if n % 500 == 0:
                    print(f"Iteration [{n + 1}/{n}], Loss: {smooth_loss:.4f}")

                n += 1
            avg_loss = epoch_loss / num_chunks
            print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
        return best_loss, best_model_state_dict






