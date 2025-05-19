import copy
from model_lstm_torch import LSTM
from DataProcessing import *
from tqdm import tqdm
import contextlib
import io


def grid_search_lstm(X_train, X_val, unique_chars, learning_rates, seq_lengths, batch_sizes, num_epochs=10, hidden_size=128):
    results = []
    best_val_loss = float('inf')
    best_model_state = None
    best_lr = None
    best_batch_size = None
    best_seq_len = None
    #X_train, X_val = TrainValTestSplit(X_input, 0.2)
    
    with open("grid_search_lstm.txt", "a") as f:
        f.write(f"\n-------Epochs: {num_epochs}-------\n")

    i = 0

    for lr in  tqdm(learning_rates, desc="LR loop"):
        for seq_len in seq_lengths:
            
            for batch_size in batch_sizes:
                #print(f"evaluating learning rate: {lr}, hidden dim: {dim} and dropout: {dropout}")
                print("Batch value: " + str(batch_size))
                X_train_batches, Y_train_batches = GetBatches(X_train.clone(), seq_len, batch_size)
                X_val_batches, Y_val_batches = GetBatches(X_val.clone(), seq_len, batch_size)
                model = LSTM(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=X_train.shape[1], num_layers=2, unique_chars = unique_chars, dropout=0.2, batch_size=batch_size, seq_len=seq_len)
                with contextlib.redirect_stdout(io.StringIO()): # Simply to block prints from the train_model function.
                    loss, model_state, epochs, loss_train, loss_val = model.train_model(
                        X_train_batches,
                        Y_train_batches,
                        X_val_batches,
                        Y_val_batches,
                        num_epochs=num_epochs,
                        learning_rate=lr,
                        best_loss_ever = 10000 # doesn't matter in the search
                    )

                min_val_loss =  min(loss_val)
                
                with open("grid_search_lstm.txt", "a") as f:
                    f.write(f"Sequence Length: {seq_len}, Learning rate: {lr}, Batch size: {batch_size}, Validation Loss: {min_val_loss}\n")

                if min_val_loss < best_val_loss:
                    best_val_loss = min_val_loss
                    best_lr = lr
                    best_seq_len = seq_len
                    best_batch_size = batch_size
                    best_model_state = model_state
                i += 1
                #print(f"{i}/{total_iteration} iterations complete!")
        #print("One dropout loop complete")

    print(f"best learning rate, dim and batch size combo: {best_lr}, {best_seq_len}, {best_batch_size}, loss: {best_val_loss:.4f}")
    return best_lr, best_seq_len, best_model_state, results
