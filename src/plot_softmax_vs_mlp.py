"""
Confronto diretto della curva di Val Accuracy tra Softmax e MLP.
Comparabili perche' entrambi calcolano la stessa identica metrica
(accuracy multiclasse: 1 su 43 classi corretta), a differenza dell'OvR
(la sua val_acc salvata per epoca e' la media di 43 accuracy binarie,
gonfiata dall'accuracy paradox - non e' la stessa quantita', va esclusa
da questo confronto).
"""
import os
import json
import matplotlib.pyplot as plt

SOFTMAX_GRID = "../results/models/grid_search_results.json"
MLP_GRID = "../results/models_mlp/grid_search_results.json"
MEDIA_DIR = "../media"


def find_best_softmax_history(all_results):
    best_name, best_hist = min(
        all_results.items(),
        key=lambda item: min(item[1]['softmax']['val_loss'])
    )
    return best_name, best_hist['softmax']


if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)

    with open(SOFTMAX_GRID) as f:
        softmax_grid = json.load(f)
    with open(MLP_GRID) as f:
        mlp_grid = json.load(f)

    softmax_name, softmax_hist = find_best_softmax_history(softmax_grid)
    mlp_name = mlp_grid['best_config']
    mlp_hist = mlp_grid['results'][mlp_name]['history']

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(range(1, len(softmax_hist['val_acc']) + 1), softmax_hist['val_acc'],
            label=f'Softmax ({softmax_name})', marker='o', markersize=3)
    ax.plot(range(1, len(mlp_hist['val_acc']) + 1), mlp_hist['val_acc'],
            label=f'MLP ({mlp_name})', marker='s', markersize=3)

    ax.set_xlabel('Epoca')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Softmax vs MLP: Val Accuracy per epoca\n(comparabile: stessa metrica, entrambi CrossEntropyLoss multiclasse)')
    ax.legend()

    out_path = os.path.join(MEDIA_DIR, "softmax_vs_mlp_val_acc.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")
