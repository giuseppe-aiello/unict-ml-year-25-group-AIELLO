"""
Demo GTSRB - Traffic Sign Recognition
======================================
Interfaccia grafica (Tkinter) per testare a mano gli 8 modelli del progetto.

Uso:
    python src/demo.py
    (oppure: cd src && python demo.py)

Permette di:
- scegliere uno degli 8 modelli da un menu a tendina;
- caricare un'immagine dal test set ufficiale (ground truth nota, con ROI
  ritagliata esattamente come in training) oppure un'immagine qualsiasi dal
  disco (ground truth sconosciuta, nessun ritaglio ROI disponibile);
- vedere l'immagine caricata, la classe vera (se nota) e la classe predetta
  dal modello selezionato, con le relative icone da data/Meta.
"""

import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
import torch
import joblib
from PIL import Image, ImageTk

from models import SoftmaxClassifier, MLPClassifier, LogisticRegression
from feature_extraction import get_feature_extractor, get_transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TEST_CSV = os.path.join(DATA_DIR, "Test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "Test")
META_DIR = os.path.join(DATA_DIR, "Meta")

NUM_CLASSES = 43
IN_FEATURES = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Percorsi delle configurazioni vincenti (vedi docs/report.md, sezione Esperimenti)
MODEL_INFO = {
    "Softmax": {
        "kind": "torch_single",
        "path": os.path.join(ROOT_DIR, "results", "models",
                              "exp_lr_0.001_mom_0.99_wd_0.0_bs_64", "softmax", "softmax_model.pth"),
        "build": lambda: SoftmaxClassifier(IN_FEATURES, NUM_CLASSES),
    },
    "One-vs-Rest (OvR)": {
        "kind": "torch_ovr",
        "dir": os.path.join(ROOT_DIR, "results", "models",
                             "exp_lr_0.001_mom_0.9_wd_0.0_bs_64", "ovr"),
    },
    "MLP": {
        "kind": "torch_single",
        "path": os.path.join(ROOT_DIR, "results", "models",
                              "mlp_h256_lr0.01_mom0.9_drop0.3", "mlp_model.pth"),
        "build": lambda: MLPClassifier(IN_FEATURES, NUM_CLASSES, hidden_size=256, dropout=0.3),
    },
    "Decision Tree": {
        "kind": "sklearn",
        "path": os.path.join(ROOT_DIR, "results", "models_classical", "decision_tree", "best_model.joblib"),
    },
    "Random Forest": {
        "kind": "sklearn",
        "path": os.path.join(ROOT_DIR, "results", "models_classical", "random_forest", "best_model.joblib"),
    },
    "SVM Lineare": {
        "kind": "sklearn",
        "path": os.path.join(ROOT_DIR, "results", "models_classical", "svm", "linear_best_model.joblib"),
    },
    "SVM RBF": {
        "kind": "sklearn",
        "path": os.path.join(ROOT_DIR, "results", "models_classical", "svm", "rbf_best_model.joblib"),
    },
    "AdaBoost": {
        "kind": "sklearn",
        "path": os.path.join(ROOT_DIR, "results", "models_classical", "adaboost", "best_model.joblib"),
    },
}

# Risultati numerici delle config vincenti, presi da docs/report.md
# (sezione Esperimenti -> "Riepilogo: confronto di tutti gli otto modelli").
# Sono valori gia' misurati durante la grid search / valutazione finale sul test
# ufficiale: qui si mostrano solo come riferimento, non vengono ricalcolati a runtime.
MODEL_METRICS = {
    "Softmax":            {"val_acc": 95.27, "test_acc": 86.63},
    "One-vs-Rest (OvR)":  {"val_acc": 98.67, "test_acc": 85.39},
    "MLP":                {"val_acc": 96.77, "test_acc": 88.12},
    "Decision Tree":      {"val_acc": 63.27, "test_acc": 50.22},
    "Random Forest":      {"val_acc": 92.29, "test_acc": 77.17},
    "SVM Lineare":        {"val_acc": 96.12, "test_acc": 87.47},
    "SVM RBF":            {"val_acc": 98.51, "test_acc": 84.85},
    "AdaBoost":           {"val_acc": 52.19, "test_acc": 45.21},
}


class InferenceEngine:
    """Carica ResNet18 (subito) e gli 8 modelli (uno alla volta, al bisogno)."""

    def __init__(self):
        self.feature_extractor = get_feature_extractor(DEVICE)
        self.transform = get_transforms()
        self._cache = {}

    def _load_model(self, name):
        if name in self._cache:
            return self._cache[name]

        info = MODEL_INFO[name]

        if info["kind"] == "sklearn":
            if not os.path.exists(info["path"]):
                raise FileNotFoundError(f"Modello non trovato: {info['path']}")
            model = joblib.load(info["path"])

        elif info["kind"] == "torch_single":
            if not os.path.exists(info["path"]):
                raise FileNotFoundError(f"Modello non trovato: {info['path']}")
            model = info["build"]()
            state = torch.load(info["path"], map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            model.to(DEVICE).eval()

        elif info["kind"] == "torch_ovr":
            classifiers = []
            for i in range(NUM_CLASSES):
                path = os.path.join(info["dir"], f"logistic_class_{i}.pth")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Modello OvR mancante per la classe {i}: {path}")
                clf = LogisticRegression(IN_FEATURES)
                state = torch.load(path, map_location=DEVICE, weights_only=True)
                clf.load_state_dict(state)
                clf.to(DEVICE).eval()
                classifiers.append(clf)
            model = classifiers

        else:
            raise ValueError(f"Tipo di modello sconosciuto: {info['kind']}")

        self._cache[name] = model
        return model

    def extract_embedding(self, pil_image):
        tensor = self.transform(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats = self.feature_extractor(tensor)
            feats = torch.flatten(feats, 1)  # [1, 512]
        return feats

    def predict(self, pil_image, model_name):
        """Ritorna (classe_predetta, confidenza, vettore_punteggi_43_classi).
        Confidenza e vettore sono None quando il modello non espone una probabilita'
        (es. LinearSVC senza probability=True)."""
        embedding = self.extract_embedding(pil_image)
        model = self._load_model(model_name)
        kind = MODEL_INFO[model_name]["kind"]
        confidence = None
        scores_vec = None

        if kind == "sklearn":
            emb_np = embedding.cpu().numpy()
            pred = int(model.predict(emb_np)[0])
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(emb_np)[0]
                    confidence = float(proba.max())
                    scores_vec = proba
                except Exception:
                    confidence = None

        elif kind == "torch_single":
            with torch.no_grad():
                logits = model(embedding)
                probs = torch.softmax(logits, dim=1)
                pred = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs.max().item())
                scores_vec = probs.cpu().numpy()[0]

        elif kind == "torch_ovr":
            with torch.no_grad():
                scores = torch.cat([torch.sigmoid(clf(embedding)) for clf in model], dim=1)  # [1, 43]
                pred = int(torch.argmax(scores, dim=1).item())
                confidence = float(scores.max().item())
                scores_vec = scores.cpu().numpy()[0]

        else:
            raise ValueError(f"Tipo di modello sconosciuto: {kind}")

        return pred, confidence, scores_vec


class DemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Demo GTSRB - Traffic Sign Recognition")
        self.root.geometry("780x620")
        self.root.resizable(False, False)

        self.engine = None
        self.current_image = None   # immagine PIL gia' ritagliata (se ROI nota), pronta per il modello
        self.current_gt = None      # ClassId vero, se noto

        self.test_df = None
        if os.path.exists(TEST_CSV):
            self.test_df = pd.read_csv(TEST_CSV)
            self.test_df["basename"] = self.test_df["Path"].apply(os.path.basename)

        self._build_ui()

        self.status_var.set("Caricamento ResNet18 in corso, attendere...")
        self.root.after(200, self._init_engine)

    # ---------- UI ----------

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Modello:").pack(side="left")
        self.model_var = tk.StringVar(value=list(MODEL_INFO.keys())[0])
        self.model_combo = ttk.Combobox(
            top, textvariable=self.model_var, values=list(MODEL_INFO.keys()),
            state="readonly", width=22
        )
        self.model_combo.pack(side="left", padx=8)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        ttk.Button(top, text="Carica da Test Set", command=self._load_from_test_set).pack(side="left", padx=4)
        ttk.Button(top, text="Carica immagine libera", command=self._load_free_image).pack(side="left", padx=4)

        stats_frame = ttk.LabelFrame(self.root, text="Statistiche del modello selezionato (da grid search / test ufficiale)", padding=8)
        stats_frame.pack(fill="x", padx=10)
        self.stats_var = tk.StringVar(value="-")
        ttk.Label(stats_frame, textvariable=self.stats_var).pack(anchor="w")
        self._update_model_stats()

        body = ttk.Frame(self.root, padding=10)
        body.pack(fill="both", expand=True)

        input_frame = ttk.LabelFrame(body, text="Input", padding=8)
        input_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.input_label = ttk.Label(input_frame)
        self.input_label.pack(expand=True)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        gt_frame = ttk.LabelFrame(right, text="Ground Truth", padding=8)
        gt_frame.pack(fill="both", expand=True, pady=(0, 8))
        self.gt_icon_label = ttk.Label(gt_frame)
        self.gt_icon_label.pack()
        self.gt_text_label = ttk.Label(gt_frame, text="-", font=("Segoe UI", 12, "bold"))
        self.gt_text_label.pack()

        pred_frame = ttk.LabelFrame(right, text="Predizione del modello", padding=8)
        pred_frame.pack(fill="both", expand=True)
        self.pred_icon_label = ttk.Label(pred_frame)
        self.pred_icon_label.pack()
        self.pred_text_label = ttk.Label(pred_frame, text="-", font=("Segoe UI", 12, "bold"))
        self.pred_text_label.pack()
        self.esito_label = ttk.Label(pred_frame, text="-", font=("Segoe UI", 10, "bold"))
        self.esito_label.pack(pady=(4, 0))
        self.time_label = ttk.Label(pred_frame, text="-")
        self.time_label.pack()
        self.top3_label = ttk.Label(pred_frame, text="-", justify="left")
        self.top3_label.pack(pady=(4, 0))

        self.status_var = tk.StringVar(value="Avvio...")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", side="bottom")

    def _on_model_change(self, event=None):
        self._update_model_stats()
        self._run_inference()

    def _update_model_stats(self):
        name = self.model_var.get()
        metrics = MODEL_METRICS.get(name)
        if metrics is None:
            self.stats_var.set("-")
            return
        gap = metrics["test_acc"] - metrics["val_acc"]
        self.stats_var.set(
            f"Val Accuracy: {metrics['val_acc']:.2f}%   |   "
            f"Test Accuracy: {metrics['test_acc']:.2f}%   |   "
            f"Gap val->test: {gap:+.2f}pt"
        )

    def _init_engine(self):
        try:
            self.engine = InferenceEngine()
            self.status_var.set("Pronto. Seleziona un modello e carica un'immagine.")
        except Exception as e:
            self.status_var.set(f"Errore caricamento ResNet18: {e}")
            messagebox.showerror("Errore", f"Impossibile caricare ResNet18:\n{e}")

    # ---------- Caricamento immagini ----------

    def _load_from_test_set(self):
        path = filedialog.askopenfilename(
            title="Scegli un'immagine dal test set ufficiale",
            initialdir=TEST_IMG_DIR if os.path.isdir(TEST_IMG_DIR) else DATA_DIR,
            filetypes=[("Immagini PNG", "*.png")]
        )
        if path:
            self._set_image(path, from_test_set=True)

    def _load_free_image(self):
        path = filedialog.askopenfilename(
            title="Scegli un'immagine qualsiasi",
            filetypes=[("Immagini", "*.png *.jpg *.jpeg *.bmp")]
        )
        if path:
            self._set_image(path, from_test_set=False)

    def _set_image(self, path, from_test_set):
        try:
            raw_image = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire l'immagine:\n{e}")
            return

        roi = None
        self.current_gt = None

        if from_test_set and self.test_df is not None:
            match = self.test_df[self.test_df["basename"] == os.path.basename(path)]
            if not match.empty:
                row = match.iloc[0]
                roi = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))
                self.current_gt = int(row["ClassId"])

        # Stesso ritaglio ROI usato in training/valutazione (vedi src/feature_extraction.py::get_sample).
        # Per un'immagine libera non c'e' ROI nota: si passa l'immagine intera (limite noto, da
        # descrivere nella sezione Demo della relazione).
        self.current_image = raw_image.crop(roi) if roi is not None else raw_image

        self._show_thumbnail(self.current_image, self.input_label, size=(220, 220))
        self._show_ground_truth()
        self._run_inference()

    # ---------- Inferenza e visualizzazione ----------

    def _show_ground_truth(self):
        if self.current_gt is None:
            self.gt_text_label.config(text="Sconosciuta\n(immagine non da test set)")
            self.gt_icon_label.config(image="")
            self.gt_icon_label.image = None
        else:
            self.gt_text_label.config(text=f"Classe {self.current_gt}")
            self._show_class_icon(self.current_gt, self.gt_icon_label)

    def _run_inference(self):
        if self.current_image is None:
            return
        if self.engine is None:
            self.status_var.set("Modello ResNet18 non ancora pronto, attendere...")
            return

        model_name = self.model_var.get()
        self.status_var.set(f"Inferenza con {model_name} in corso...")
        self.root.update_idletasks()

        t0 = time.perf_counter()
        try:
            pred, confidence, scores_vec = self.engine.predict(self.current_image, model_name)
        except Exception as e:
            self.status_var.set(f"Errore durante l'inferenza con {model_name}: {e}")
            messagebox.showerror("Errore inferenza", str(e))
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000

        conf_text = f"Confidenza: {confidence * 100:.1f}%" if confidence is not None else "Confidenza: N/D"
        self.pred_text_label.config(text=f"Classe {pred}\n{conf_text}")
        self._show_class_icon(pred, self.pred_icon_label)

        self.time_label.config(text=f"Tempo di inferenza: {elapsed_ms:.1f} ms")

        # Top-3 classi con relativo punteggio, quando il modello espone una distribuzione
        if scores_vec is not None:
            top3_idx = scores_vec.argsort()[::-1][:3]
            top3_lines = "\n".join(f"{i+1}. Classe {c} ({scores_vec[c] * 100:.1f}%)" for i, c in enumerate(top3_idx))
            self.top3_label.config(text=f"Top-3:\n{top3_lines}")
        else:
            self.top3_label.config(text="Top-3: N/D per questo modello")

        if self.current_gt is not None:
            if pred == self.current_gt:
                self.esito_label.config(text="Predizione CORRETTA", foreground="dark green")
            else:
                self.esito_label.config(text="Predizione SBAGLIATA", foreground="red")
            self.status_var.set(f"Modello: {model_name} | {elapsed_ms:.1f} ms")
        else:
            self.esito_label.config(text="Ground truth sconosciuta", foreground="black")
            self.status_var.set(f"Modello: {model_name} | {elapsed_ms:.1f} ms")

    def _show_class_icon(self, class_id, label_widget):
        icon_path = os.path.join(META_DIR, f"{class_id}.png")
        if os.path.exists(icon_path):
            icon = Image.open(icon_path).convert("RGB")
            self._show_thumbnail(icon, label_widget, size=(64, 64))
        else:
            label_widget.config(image="")
            label_widget.image = None

    @staticmethod
    def _show_thumbnail(pil_image, label_widget, size):
        thumb = pil_image.copy()
        thumb.thumbnail(size)
        photo = ImageTk.PhotoImage(thumb)
        label_widget.config(image=photo)
        label_widget.image = photo  # riferimento vivo, altrimenti Tkinter lo garbage-collecta


def main():
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
