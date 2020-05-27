def suggest_lr(trial):
    trial.suggest_loguniform("lr", 1e-4, 1e-1)


def suggest_hidden_sizes(trial, max_num_layers=3):
    num_layers = trial.suggest_int("num_layers", 2, max_num_layers)
    for k in range(num_layers):
        trial.suggest_int(f"hidden_sizes:{k}", 10, 30)
