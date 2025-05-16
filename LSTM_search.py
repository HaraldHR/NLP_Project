import copy
from model_lstm_torch import LSTM
from DataProcessing import GetTrainBatches
from tqdm import tqdm

def grid_search_lstm(X_input, unique_chars, learning_rates, hidden_dims, batch_sizes, num_epochs=10, seq_len=50):
    results = []
    best_loss = float('inf')
    best_model_state = None
    best_lr = None
    best_dim = None
    best_batch_size = None

    total_iteration = len(learning_rates) * len(hidden_dims) * len(batch_sizes)
    i = 0
    for lr in  tqdm(learning_rates, desc="LR loop"):
        for dim in tqdm(hidden_dims, desc="Hidden dims loop"):
            for batch_size in tqdm(batch_sizes, desc="Batch size loop"):
                #print(f"evaluating learning rate: {lr}, hidden dim: {dim} and dropout: {dropout}")
                X_batches, Y_batches = GetTrainBatches(X_input.clone(), seq_len, batch_size)
                model = LSTM(input_size=X_input.shape[1], hidden_size=dim, output_size=X_input.shape[1], num_layers=2, unique_chars = unique_chars, dropout=0.2, batch_size=batch_size, seq_len=seq_len)

                # väldigt anpassat till min RNN klass just nu så kanske behöver skrivas om
                #för att passa mer generellt
                loss, model_state = model.train_model(
                    X_batches,
                    Y_batches,
                    num_epochs=num_epochs,
                    learning_rate=lr,
                    best_loss_ever = best_loss # doesn't matter in the search
                )

                results.append((dim, lr, batch_size, loss))

                if loss < best_loss:
                    best_loss = loss
                    best_lr = lr
                    best_dim = dim
                    best_batch_size = batch_size
                    best_model_state = model_state
                i += 1
                #print(f"{i}/{total_iteration} iterations complete!")
        #print("One dropout loop complete")

    print(f"best learning rate, dim and batch size combo: {best_lr}, {best_dim}, {best_batch_size}, loss: {best_loss:.4f}")
    with open("grid_search_lstm.txt", "a") as f:
        f.write(f"\n-------Epochs: {num_epochs}-------\n")
        for result in results:
            f.write(f"Hidden dim: {result[0]}, Learning rate: {result[1]}, Batch size: {result[2]}, Loss: {result[3]}\n")
    return best_lr, best_dim, best_model_state, results
