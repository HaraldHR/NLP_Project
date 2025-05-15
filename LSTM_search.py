import copy
from model_lstm_torch import LSTM
from tqdm import tqdm

def grid_search_lstm(X_input, unique_chars, learning_rates, hidden_dims, dropout_values, num_epochs=1, seq_len=50):
    results = []
    best_loss = float('inf')
    best_model_state = None
    best_lr = None
    best_dim = None

    total_iteration = len(learning_rates) * len(hidden_dims) * len(dropout_values)
    i = 0
    for lr in  tqdm(learning_rates, desc="LR loop"):
        for dim in tqdm(hidden_dims, desc="Hidden dims loop"):
            #print(f"evaluating learning rate: {lr}, hidden dim: {dim} and dropout: {dropout}")
            model = LSTM(input_size=X_input.shape[1], hidden_size=dim, output_size=X_input.shape[1], num_layers=2, unique_chars = unique_chars, dropout=dropout)
            # har inte testat att detta funkar än, men hoppas

            # väldigt anpassat till min RNN klass just nu så kanske behöver skrivas om
            #för att passa mer generellt
            loss, model_state = model.train_model(
                X_input.clone(),
                num_epochs=num_epochs,
                seq_len=seq_len,
                learning_rate=lr,
                best_loss_ever = best_loss # doesn't matter in the search
            )

            results.append((dim, lr, loss))

            if loss < best_loss:
                best_loss = loss
                best_lr = lr
                best_dim = dim
                best_model_state = model_state
            i += 1
            #print(f"{i}/{total_iteration} iterations complete!")
        #print("One dropout loop complete")

    print(f"best learning  rate and dim combo: {best_lr}, {best_dim}, loss: {best_loss:.4f}")
    with open("grid_search_lstm.txt", "a") as f:
        f.write(f"\n-------Epochs: {num_epochs}-------\n")
        for result in results:
            f.write(f"Hidden dim: {result[0]}, Learning rate: {result[1]}, Loss: {result[2]}\n")
    return best_lr, best_dim, best_model_state, results
