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
                epoch_loss += loss.item()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    #torch.save(self.state_dict(), best_model_path)
                    best_model_state_dict = self.state_dict()
                    print(f"Best model saved with loss: {best_loss:.4f}")
                if c % 500 == 0:
                    print(f"Chunk [{c + 1}/{num_chunks}], Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / num_chunks
            print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
        return best_loss, best_model_state_dict







class ManualRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.m = hidden_size
        self.K = output_size
        self.input_size = input_size

        # Initialize weights like in NumPy version
        self.params = {
            "U": torch.randn(hidden_size, input_size, requires_grad=True),
            "W": torch.randn(hidden_size, hidden_size, requires_grad=True),
            "V": torch.randn(output_size, hidden_size, requires_grad=True),
            "b": torch.zeros(hidden_size, 1, requires_grad=True),
            "c": torch.zeros(output_size, 1, requires_grad=True)
        }

    def forward_pass(self, X, Y):
        seq_length = X.shape[1]
        batch_size = X.shape[0]

        a, h, o, p = {}, {}, {}, {}

        h[-1] = torch.zeros(self.m, batch_size)

        loss = 0

        for t in range(seq_length):
            x_t = X[:, t, :].T  # (input_size, batch)
            y_t = Y[:, t, :].T  # (output_size, batch)

            a[t] = self.params["W"] @ h[t-1] + self.params["U"] @ x_t + self.params["b"]
            h[t] = torch.tanh(a[t])
            o[t] = self.params["V"] @ h[t] + self.params["c"]
            p[t] = F.softmax(o[t], dim=0)

            loss += -torch.sum(torch.log(torch.sum(y_t * p[t], dim=0) + 1e-12))

        loss = loss.item() / seq_length

        outputs = {"a": a, "h": h, "o": o, "p": p}
        return loss, outputs

    def backward_pass(self, X, Y, outputs):
        seq_length = X.shape[1]
        batch_size = X.shape[0]

        grads = {k: torch.zeros_like(v) for k, v in self.params.items()}
        a, h, o, p = outputs["a"], outputs["h"], outputs["o"], outputs["p"]

        dh_next = torch.zeros(self.m, batch_size)

        for t in reversed(range(seq_length)):
            x_t = X[:, t, :].T
            y_t = Y[:, t, :].T

            do = p[t] - y_t
            grads["c"] += do.sum(dim=1, keepdim=True)
            grads["V"] += do @ h[t].T

            dh = self.params["V"].T @ do + dh_next
            da = dh * (1 - torch.tanh(a[t]) ** 2)

            grads["b"] += da.sum(dim=1, keepdim=True)
            grads["W"] += da @ h[t - 1].T
            grads["U"] += da @ x_t.T

            dh_next = self.params["W"].T @ da

        # Average gradients over time
        for k in grads:
            grads[k] /= seq_length

        return grads

    import torch
    import numpy as np
    import torch.nn.functional as F

    def one_hot_encode(self, char_index, vocab_size):
        vec = torch.zeros(vocab_size, 1)
        vec[char_index] = 1.0
        return vec

    def synthesize_seq(self, rnn, h0, x0, length, ind_to_char):
        """Generates a sequence of characters from trained RNN."""
        h = h0.clone()
        x = x0.clone()
        result = []

        for t in range(length):
            a = rnn.params["W"] @ h + rnn.params["U"] @ x + rnn.params["b"]
            h = torch.tanh(a)
            o = rnn.params["V"] @ h + rnn.params["c"]
            p = F.softmax(o, dim=0)

            idx = torch.multinomial(p.view(-1), 1).item()
            x = self.one_hot_encode(idx, p.size(0))
            result.append(ind_to_char[idx])

        return ''.join(result)

    def train_rnn(self, rnn, book_data, char_to_ind, ind_to_char, num_epochs=3, seq_length=25, eta=0.001):
        m = rnn.m
        K = rnn.K
        h0 = torch.zeros(m, 1)

        # Adam state
        m_params = {k: torch.zeros_like(v) for k, v in rnn.params.items()}
        v_params = {k: torch.zeros_like(v) for k, v in rnn.params.items()}
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        n = 0
        loss_history = []
        best_loss = float('inf')
        best_params = None

        for epoch in range(num_epochs):
            e = 0
            hprev = h0.clone()

            while e + seq_length + 1 < len(book_data):
                X_chars = book_data[e:e + seq_length]
                Y_chars = book_data[e + 1:e + seq_length + 1]

                # One-hot encode input/output
                X = torch.zeros(K, seq_length)
                Y = torch.zeros(K, seq_length)
                for t in range(seq_length):
                    X[char_to_ind[X_chars[t]], t] = 1
                    Y[char_to_ind[Y_chars[t]], t] = 1

                X = X.unsqueeze(0)  # shape: (1, seq_len, K)
                Y = Y.unsqueeze(0)

                loss, outputs = rnn.forward_pass(X, Y)
                grads = rnn.backward_pass(X, Y, outputs)

                # Clip gradients
                for k in grads:
                    grads[k] = grads[k].clamp(-5, 5)

                # Smooth loss
                if n == 0:
                    smooth_loss = loss
                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss

                if smooth_loss < best_loss:
                    best_loss = smooth_loss
                    best_params = {k: v.clone() for k, v in rnn.params.items()}

                # Adam update
                for k in rnn.params:
                    m_params[k] = beta1 * m_params[k] + (1 - beta1) * grads[k]
                    v_params[k] = beta2 * v_params[k] + (1 - beta2) * (grads[k] ** 2)

                    m_hat = m_params[k] / (1 - beta1 ** (n + 1))
                    v_hat = v_params[k] / (1 - beta2 ** (n + 1))

                    rnn.params[k] = rnn.params[k] - eta * m_hat / (torch.sqrt(v_hat) + epsilon)

                loss_history.append(smooth_loss)

                # Print and sample
                if n == 0 or n % 5000 == 0:
                    print(f"Iteration {n}, smooth loss: {smooth_loss:.4f}")
                    x0 = self.one_hot_encode(char_to_ind[X_chars[0]], K)
                    sample = self.synthesize_seq(rnn, hprev, x0, 200, ind_to_char)
                    print("--- text ---")
                    print(sample)
                    print("------------")

                hprev = outputs["h"][seq_length - 1].detach()
                e += seq_length
                n += 1

        return rnn, best_params, best_loss, loss_history

