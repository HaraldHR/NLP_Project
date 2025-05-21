import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import pickle
import torch.nn.functional as F


class RNN(nn.Module):


    def __init__(self, input_size, hidden_size, output_size, num_layers=2, nonlinearity='tanh'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,  # 'tanh' or 'relu'
            batch_first=False,
            dropout = 0.2# input shape (seq_len, batch, input_size)
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden=None):

        rnn_out, hidden = self.rnn(input_seq, hidden)
        # Apply linear layer to each time step
        output = self.fc(rnn_out)  # (seq_len, batch, output_size)
        return output, hidden

    def forward_loss(self, X_batches, Y_batches):
        self.eval()
        output_size = self.fc.out_features
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        hidden = None

        with torch.no_grad():
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                target_seq = torch.argmax(Y_batch, dim=2)  # shape: (seq_len, batch)
                output_seq, hidden = self.forward(X_batch, hidden)

                if hidden is not None:
                    if isinstance(hidden, tuple):
                        hidden = tuple(h.detach() for h in hidden)
                    else:
                        hidden = hidden.detach()

                output_seq = output_seq.view(-1, output_size)
                target_seq = target_seq.view(-1)

                loss = criterion(output_seq, target_seq)
                total_loss += loss.item()

        avg_loss = total_loss / len(X_batches)
        return avg_loss
    def train_model(self, X_train_batches, Y_train_batches, X_val_batches, Y_val_batches, num_epochs,
                    learning_rate=0.001, best_loss_ever=1e6):
        output_size = self.fc.out_features
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        best_model_state_dict = {}
        best_loss = best_loss_ever
        n = 0

        loss_train = []
        loss_val = []
        epochs = []

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            hidden = None

            for i in range(X_train_batches.shape[0]):
                X_batch = X_train_batches[i]  # shape: (seq_len, batch, input_size)
                Y_batch = Y_train_batches[i]  # shape: (seq_len, batch, output_size)
                target_seq = torch.argmax(Y_batch, dim=2)  # shape: (seq_len, batch)

                optimizer.zero_grad()

                output_seq, hidden = self.forward(X_batch, hidden)

                if hidden is not None:
                    hidden = hidden.detach()

                # Reshape to (seq_len * batch, output_size) and (seq_len * batch)
                output_seq = output_seq.view(-1, output_size)
                target_seq = target_seq.view(-1)

                loss = criterion(output_seq, target_seq)
                epoch_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
                optimizer.step()

                if n == 0:
                    smooth_loss_train = loss.item()
                else:
                    smooth_loss_train = 0.999 * smooth_loss_train + 0.001 * loss.item()

                """
                if smooth_loss_train < best_loss:
                    best_loss = smooth_loss_train
                    best_model_state_dict = self.state_dict()
                """
                if n % 500 == 0:
                    print(f"Iteration {n}, Smooth Loss: {smooth_loss_train:.4f}")

                n += 1
            """
            # --- Validation ---
            self.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for j in range(X_val_batches.shape[0]):
                    val_output, _ = self.forward(X_val_batches[j], None)
                    val_targets = torch.argmax(Y_val_batches[j], dim=2)
                    val_output = val_output.view(-1, output_size)
                    val_targets = val_targets.view(-1)
                    val_loss = criterion(val_output, val_targets)
                    val_loss_sum += val_loss.item()
                    if val_loss.item() < best_loss:
                        best_loss = val_loss.item()
                        best_model_state_dict = self.state_dict()
            """
            self.train()
            train_loss_avg = epoch_loss / X_train_batches.shape[0]
            best_model_state_dict = self.state_dict()
            #val_loss_avg = val_loss_sum / X_val_batches.shape[0]



            loss_train.append(train_loss_avg)
            #loss_val.append(val_loss_avg)
            epochs.append(epoch)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss_avg:.4f}")

        return loss_train, loss_val, epochs, best_loss, best_model_state_dict


    def synthesize_text(self, x0, n, ind_to_char, char_to_ind, device='cpu'):

        self.eval()
        K = len(char_to_ind)
        generated_indices = []

        # Prepare initial input x0 (e.g., one-hot vector for a character)
        x = torch.tensor(x0, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)  # (1, 1, K)
        hidden = None

        with torch.no_grad():
            for _ in range(n):
                output, hidden = self.forward(x, hidden)  # output: (1, 1, K)
                output = output.squeeze(0).squeeze(0)  # shape: (K,)
                probs = torch.softmax(output, dim=0).cpu().numpy()
                idx = np.random.choice(K, p=probs)
                generated_indices.append(idx)

                # Create new one-hot encoded input
                x = torch.zeros((1, 1, K), dtype=torch.float32).to(device)
                x[0, 0, idx] = 1.0

        self.train()
        generated_text = ''.join([ind_to_char[i] for i in generated_indices])
        return generated_text







