import torch
import torch.nn.functional as F
import numpy as np



def synthesize_text(model, x0, n, ind_to_char, char_to_ind, device='cpu'):
    model.eval()
    K = len(char_to_ind)
    generated_indices = []

    x = torch.tensor(x0, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)  # (1, 1, K)
    hidden = None

    with torch.no_grad():
        for _ in range(n):
            output, hidden = model(x, hidden)
            output = output.squeeze(0).squeeze(0)
            probs = torch.softmax(output, dim=0).cpu().numpy()

            # sample from probs
            idx = np.random.choice(K, p=probs)
            generated_indices.append(idx)

            x = torch.zeros((1, 1, K), dtype=torch.float32).to(device)
            x[0, 0, idx] = 1.0

    generated_text = ''.join([ind_to_char[i] for i in generated_indices])
    return generated_text


