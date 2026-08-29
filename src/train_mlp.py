import os
import json
from itertools import product
from training import train_mlp

MODELS_OUT_DIR = "../results/models_mlp"

if __name__ == "__main__":
    # wd e batch_size fissati ai valori gia' visti funzionare bene per Softmax
    # (wd ha impatto marginale, bs=64 sempre tra i migliori) - si esplorano
    # invece gli iperparametri nuovi/specifici dell'MLP (hidden_size, dropout)
    # insieme a lr/momentum, che nel progetto si sono gia' dimostrati sensibili
    # a seconda del modello (mom=0.99 buono per Softmax, mom=0.9 per OvR).
    hidden_sizes = [64, 128, 256]
    learning_rates = [0.01, 0.001]
    momentums = [0.9, 0.99]
    dropouts = [0.0, 0.3]
    weight_decay = 0.0
    batch_size = 64
    max_epochs = 50

    best_val_loss = float('inf')
    best_config_name = None

    all_results = {}

    for hidden_size, lr, mom, dropout in product(hidden_sizes, learning_rates, momentums, dropouts):
        config_name = f"mlp_h{hidden_size}_lr{lr}_mom{mom}_drop{dropout}"
        print(f"\n=======================================================================")
        print(f"TRAINING CONFIG: {config_name}")
        print(f"=======================================================================")

        model, hist = train_mlp(
            lr=lr,
            momentum=mom,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=max_epochs,
            hidden_size=hidden_size,
            dropout=dropout,
            subdir=config_name
        )

        all_results[config_name] = {
            'hidden_size': hidden_size,
            'lr': lr,
            'momentum': mom,
            'dropout': dropout,
            'history': hist
        }

        min_val_loss = min(hist['val_loss'])
        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_config_name = config_name

    print("\n========================================")
    print(f"RICERCA COMPLETATA. Configurazione ottimale MLP: {best_config_name}")
    print(f"Miglior Validation Loss: {best_val_loss:.4f}")
    print("========================================")

    os.makedirs(MODELS_OUT_DIR, exist_ok=True)
    out_file = os.path.join(MODELS_OUT_DIR, "grid_search_results.json")
    with open(out_file, 'w') as f:
        json.dump({'results': all_results, 'best_config': best_config_name}, f, indent=4)
    print(f"Tutte le curve sono state salvate in {out_file}.")
