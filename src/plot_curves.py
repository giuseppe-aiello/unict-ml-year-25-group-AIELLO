import os
import json
import matplotlib.pyplot as plt

MODELS_DIR = "../results/models"
GRID_SEARCH_RESULTS = os.path.join(MODELS_DIR, "grid_search_results.json")
MEDIA_DIR = "../media"


def find_best_config(all_results, key):
    best_name, best_hist = min(
        all_results.items(),
        key=lambda item: min(item[1][key]['val_loss'])
    )
    return best_name, best_hist[key]


def plot_train_val(hist, title, out_path):
    epochs = range(1, len(hist['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, hist['train_loss'], label='Train Loss')
    axes[0].plot(epochs, hist['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(epochs, hist['train_acc'], label='Train Acc')
    axes[1].plot(epochs, hist['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_ovr_instability(all_results, best_name, unstable_name, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    best_hist = all_results[best_name]['ovr_mean']
    unstable_hist = all_results[unstable_name]['ovr_mean']

    ax.plot(range(1, len(best_hist['val_loss']) + 1), best_hist['val_loss'],
            label=f'Stabile: {best_name}')
    ax.plot(range(1, len(unstable_hist['val_loss']) + 1), unstable_hist['val_loss'],
            label=f'Instabile: {unstable_name}')

    ax.set_xlabel('Epoca')
    ax.set_ylabel('Val Loss (media OvR)')
    ax.set_title('OvR: instabilità con lr=0.01 + momentum=0.99')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)

    with open(GRID_SEARCH_RESULTS) as f:
        all_results = json.load(f)

    best_softmax_name, best_softmax_hist = find_best_config(all_results, 'softmax')
    best_ovr_name, best_ovr_hist = find_best_config(all_results, 'ovr_mean')

    plot_train_val(
        best_softmax_hist,
        f"Softmax - miglior config ({best_softmax_name})",
        os.path.join(MEDIA_DIR, "curve_softmax_best.png")
    )
    plot_train_val(
        best_ovr_hist,
        f"OvR (media 43 classificatori) - miglior config ({best_ovr_name})",
        os.path.join(MEDIA_DIR, "curve_ovr_best.png")
    )

    unstable_name = "exp_lr_0.01_mom_0.99_wd_0.0001_bs_64"
    plot_ovr_instability(
        all_results, best_ovr_name, unstable_name,
        os.path.join(MEDIA_DIR, "ovr_instabilita.png")
    )
