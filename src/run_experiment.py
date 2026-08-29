import os
import json
import numpy as np
from training import train_softmax, train_logistic_ovr

def aggregate_ovr_histories(all_ovr_histories):
    max_epochs = max(len(h['train_loss']) for h in all_ovr_histories)
    
    agg_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for key in agg_history.keys():
        matrix = []
        for h in all_ovr_histories:
            curve = h[key]
            if len(curve) < max_epochs:
                curve = curve + [curve[-1]] * (max_epochs - len(curve))
            matrix.append(curve)
        
        agg_history[key] = np.mean(matrix, axis=0).tolist()
        
    return agg_history

def run_experiment(lr, momentum, weight_decay, batch_size, epochs, subdir_name):
    print(f"\n=======================================================================")
    print(f"TRAINING CONFIG: LR={lr} | MOM={momentum} | WD={weight_decay} | BATCH={batch_size}")
    print(f"=======================================================================")

    print("\n>>> 1. Training Softmax (43 Classi)")
    model_softmax, hist_softmax = train_softmax(
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs, 
        subdir=os.path.join(subdir_name, "softmax")
    )

    print(f"\n>>> 2. Training OvR Global (43 modelli binari)")
    num_classes = 43
    all_ovr_histories = []

    for i in range(num_classes):
        print(f"\n--- Classe {i}/{num_classes-1} ---")
        model_log, hist_log = train_logistic_ovr(
            target_class_id=i, 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs, 
            subdir=os.path.join(subdir_name, "ovr")
        )
        all_ovr_histories.append(hist_log)

    hist_ovr_mean = aggregate_ovr_histories(all_ovr_histories)

    return hist_softmax, hist_ovr_mean

if __name__ == "__main__":
    learning_rates = [0.01, 0.001]
    momentums = [0.9, 0.99]
    weight_decays = [0.0, 1e-4]
    batch_sizes = [64, 128]
    max_epochs = 50 
    
    best_softmax_val_loss = float('inf')
    best_softmax_config = {}
    best_ovr_val_loss = float('inf')
    best_ovr_config = {}

    all_results = {}

    for lr in learning_rates:
        for mom in momentums:
            for wd in weight_decays:
                for bs in batch_sizes:

                    config_name = f"exp_lr_{lr}_mom_{mom}_wd_{wd}_bs_{bs}"

                    hist_soft, hist_ovr_mean = run_experiment(lr, mom, wd, bs, max_epochs, config_name)

                    all_results[config_name] = {
                        'softmax': hist_soft,
                        'ovr_mean': hist_ovr_mean
                    }

                    min_softmax_val_loss = min(hist_soft['val_loss'])
                    if min_softmax_val_loss < best_softmax_val_loss:
                        best_softmax_val_loss = min_softmax_val_loss
                        best_softmax_config = {
                            'lr': lr,
                            'mom': mom,
                            'wd': wd,
                            'bs': bs,
                            'val_loss': min_softmax_val_loss
                        }

                    min_ovr_val_loss = min(hist_ovr_mean['val_loss'])
                    if min_ovr_val_loss < best_ovr_val_loss:
                        best_ovr_val_loss = min_ovr_val_loss
                        best_ovr_config = {
                            'lr': lr,
                            'mom': mom,
                            'wd': wd,
                            'bs': bs,
                            'val_loss': min_ovr_val_loss
                        }

    print("\n========================================")
    print(f"RICERCA COMPLETATA. Configurazione ottimale SOFTMAX:")
    print(f"Learning Rate: {best_softmax_config['lr']}")
    print(f"Momentum: {best_softmax_config['mom']}")
    print(f"Weight Decay: {best_softmax_config['wd']}")
    print(f"Batch Size: {best_softmax_config['bs']}")
    print(f"Miglior Validation Loss: {best_softmax_config['val_loss']:.4f}")
    print("----------------------------------------")
    print(f"Configurazione ottimale OVR:")
    print(f"Learning Rate: {best_ovr_config['lr']}")
    print(f"Momentum: {best_ovr_config['mom']}")
    print(f"Weight Decay: {best_ovr_config['wd']}")
    print(f"Batch Size: {best_ovr_config['bs']}")
    print(f"Miglior Validation Loss: {best_ovr_config['val_loss']:.4f}")
    print("========================================")

    out_file = "../results/models/grid_search_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Tutte le curve sono state salvate in {out_file}.")