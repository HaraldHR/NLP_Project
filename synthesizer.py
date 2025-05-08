import torch
import torch.nn.functional as F
import numpy as np

def synthesize_text(model, start_char, char_to_ind, ind_to_char, length, device='cpu'):
    model.eval()  # Set to evaluation mode
    K = len(char_to_ind)

    # Initialize input
    x = np.zeros((1, 1, K), dtype=np.float32)  # (seq_len=1, batch=1, input_size)
    x[0, 0, char_to_ind[start_char]] = 1
    x = torch.tensor(x, dtype=torch.float32).to(device)

    # Initialize hidden state
    hidden = torch.zeros(model.num_layers, 1, model.hidden_size).to(device)

    generated_indices = []

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(x, hidden)  # output: (1, 1, output_size)
            output = output.squeeze(0).squeeze(0)  # (output_size,)
            probs = F.softmax(output, dim=0).cpu().numpy()

            # Sample next character
            next_index = np.random.choice(len(probs), p=probs)
            generated_indices.append(next_index)

            # Prepare next input
            x = np.zeros((1, 1, K), dtype=np.float32)
            x[0, 0, next_index] = 1
            x = torch.tensor(x, dtype=torch.float32).to(device)

    # Convert indices to string
    generated_text = ''.join([ind_to_char[idx] for idx in generated_indices])
    return generated_text


