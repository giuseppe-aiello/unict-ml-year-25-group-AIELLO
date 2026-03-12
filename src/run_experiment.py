import os
from training import train_softmax, train_logistic_ovr

def run_experiment(lr, momentum, weight_decay, batch_size, epochs, subdir_name):
    print(f"\n=======================================================================")
    print(f"TRAINING CONFIG: LR={lr} | MOM={momentum} | WD={weight_decay} | BATCH={batch_size}")
    print(f"=======================================================================")

    print("\n>>> 1. Training Softmax (43 Classi)")
    model_softmax, hist_softmax = train_softmax(
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay,
        epochs=epochs, 
        batch_size=batch_size,
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
            epochs=epochs, 
            batch_size=batch_size,
            subdir=os.path.join(subdir_name, "ovr")
        )
        all_ovr_histories.append(hist_log)

    return hist_softmax, all_ovr_histories

if __name__ == "__main__":
    # Spazio di ricerca espanso
    learning_rates = [0.01, 0.001]
    momentums = [0.9, 0.99]
    weight_decays = [0.0, 1e-4]
    batch_sizes = [64, 128]
    max_epochs = 50 
    
    best_val_loss = float('inf')
    best_config = {}

    for lr in learning_rates:
        for mom in momentums:
            for wd in weight_decays:
                for bs in batch_sizes:
                    
                    subdir = f"exp_lr_{lr}_mom_{mom}_wd_{wd}_bs_{bs}"
                    
                    hist_soft, hist_ovr = run_experiment(lr, mom, wd, bs, max_epochs, subdir)
                    
                    min_val_loss = min(hist_soft['val_loss'])
                    
                    if min_val_loss < best_val_loss:
                        best_val_loss = min_val_loss
                        best_config = {
                            'lr': lr, 
                            'mom': mom, 
                            'wd': wd, 
                            'bs': bs, 
                            'val_loss': min_val_loss
                        }
    
    print("\n========================================")
    print(f"RICERCA COMPLETATA. Configurazione ottimale:")
    print(f"Learning Rate: {best_config['lr']}")
    print(f"Momentum: {best_config['mom']}")
    print(f"Weight Decay: {best_config['wd']}")
    print(f"Batch Size: {best_config['bs']}")
    print(f"Miglior Validation Loss: {best_config['val_loss']:.4f}")
    print("========================================")