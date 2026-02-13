import os
from training import train_softmax, train_logistic_ovr


def run_train_exp1():

    print("\n========================================")
    print("FASE DI TRAINING PER EXP N.1")
    print("========================================")


    LR = 0.001
    MOM = 0.9
    EPOCHS_SOFTMAX = 15
    EPOCHS_OVR = 15
    SUBDIR= "exp1" #sotto cartella dove salverò i modelli allenati

    print("\n>>> 1. Training Softmax (43 Classi)")
    model_softmax, hist_softmax = train_softmax(
        lr=LR, 
        momentum=MOM, 
        epochs=EPOCHS_SOFTMAX, 
        subdir=os.path.join(SUBDIR, "softmax")
    )

    # --- 2. TRAINING OVR (43 modelli binari indipendenti) ---
    print(f"\n>>> 2. Training OvR Global (43 modelli binari)")
    num_classes = 43
    all_ovr_histories = []

    for i in range(num_classes):
        print(f"\n--- Classe {i}/{num_classes-1} ---")
        # Salvataggio in: results/models/exp_full_.../ovr/
        model_log, hist_log = train_logistic_ovr(
            target_class_id=i, 
            lr=LR, 
            momentum=MOM, 
            epochs=EPOCHS_OVR, 
            subdir=os.path.join(SUBDIR, "ovr")
        )
        all_ovr_histories.append(hist_log)

    return hist_softmax, all_ovr_histories

if __name__ == "__main__":
    run_train_exp1()