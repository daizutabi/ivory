def suggest_lr(trial, min, max):
    trial.suggest_loguniform("lr", min, max)


def suggest_hidden_sizes(trial, max_num_layers, min_size=10, max_size=30):
    num_layers = trial.suggest_int("num_layers", 2, max_num_layers)
    for k in range(num_layers):
        trial.suggest_int(f"hidden_sizes:{k}", min_size, max_size)
