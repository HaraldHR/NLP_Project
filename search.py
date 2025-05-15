import copy

def grid_search_lr(model, X_input, Y_input, learning_rates, num_epochs=5, seq_len=50):
    results = []
    best_loss = float('inf')
    best_model_state = None
    best_lr = None

    for lr in learning_rates:
        print(f"evaluating learning rate: {lr}")

        # har inte testat att detta funkar än, men hoppas
        model_copy = copy.deepcopy(model)

        # väldigt anpassat till min RNN klass just nu så kanske behöver skrivas om
        #för att passa mer generellt
        loss, model_state = model_copy.train_model(
            X_input.clone(), Y_input.clone(),
            num_epochs=num_epochs,
            seq_len=seq_len,
            learning_rate=lr
        )

        results.append((lr, loss))

        if loss < best_loss:
            best_loss = loss
            best_lr = lr
            best_model_state = model_state

    print(f"best learning  rate: {best_lr}, loss: {best_loss:.4f}")
    return best_lr, best_model_state, results