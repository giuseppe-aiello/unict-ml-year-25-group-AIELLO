import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = "../data"
MEDIA_DIR = "../media"


def plot_class_distribution(train_df, test_df, out_path):
    train_counts = train_df['ClassId'].value_counts().sort_index()
    test_counts = test_df['ClassId'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    x = train_counts.index
    width = 0.4
    ax.bar(x - width / 2, train_counts.values, width, label='Train (39.209 immagini)')
    ax.bar(x + width / 2, test_counts.values, width, label='Test (12.630 immagini)')

    ax.set_xlabel('ClassId (0-42)')
    ax.set_ylabel('Numero di immagini')
    ax.set_title('GTSRB: distribuzione delle immagini per classe (train vs test)')
    ax.set_xticks(x)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_class_examples(train_df, out_path):
    fig, axes = plt.subplots(5, 9, figsize=(16, 9))
    axes = axes.flatten()

    for class_id in range(43):
        row = train_df[train_df['ClassId'] == class_id].iloc[0]
        img_path = os.path.join(DATA_DIR, row['Path'])
        img = Image.open(img_path)

        ax = axes[class_id]
        ax.imshow(img)
        ax.set_title(str(class_id), fontsize=9)
        ax.axis('off')

    for ax in axes[43:]:
        ax.axis('off')

    fig.suptitle('GTSRB: un esempio reale per ciascuna delle 43 classi (immagini originali, non ridimensionate)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))

    plot_class_distribution(train_df, test_df, os.path.join(MEDIA_DIR, "dataset_class_distribution.png"))
    plot_class_examples(train_df, os.path.join(MEDIA_DIR, "dataset_class_examples.png"))
