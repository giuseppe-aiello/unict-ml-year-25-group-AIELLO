import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split

# Transformations for input images - Standardizzo le dimensioni e proprietà
def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Load the GTSRB CSV as a DataFrame
def load_gtsrb_dataframe(csv_file):
    return pd.read_csv(csv_file)

# Load a single sample (image + label)
def get_sample(df, idx, root_dir, transform=None, use_roi=True):
    row = df.iloc[idx] #prendo riga del datafrabe


    img_path = os.path.join(root_dir, row["Path"])

    image = Image.open(img_path).convert("RGB")

    label = int(row["ClassId"])

    if use_roi: #ritagli la ROI basato sull'annotazione  del singolo esempio
        x1, y1, x2, y2 = int(row["Roi.X1"]), int(row["Roi.Y1"]), \
                         int(row["Roi.X2"]), int(row["Roi.Y2"])
        image = image.crop((x1, y1, x2, y2))

    if transform is not None:
        image = transform(image)

    return image, label

# Functional Dataset wrapping the get_sample function
class FunctionalDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, use_roi=True):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.use_roi = use_roi

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return get_sample(
            self.df, idx, #numero dell'immagine
            root_dir=self.root_dir,
            transform=self.transform,
            use_roi=self.use_roi
        )


# Prepare a ResNet18 feature extractor
def get_feature_extractor(device):
    print("Caricamento modello ResNet18 pre-addestrato...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
    extractor = torch.nn.Sequential(*list(resnet.children())[:-1]) #tutti componenti tranne fc ovvero classificatore finale quindi
    ##solo feature, no class finali
    return extractor.to(device).eval()

# Extract embeddings from all images in the DataLoader
def extract_embeddings(loader, feature_extractor, device):
    X_list, y_list = [], []

    for xb, yb in tqdm(loader, desc="Extracting features"):
        #xb: Contiene il batch di immagini
        #yb: Contiene le relative etichette
        with torch.no_grad():
            xb = xb.to(device)
            feats = feature_extractor(xb)   # Output: [Batch, 512, 1, 1] APPENA DOPO FEATURE EXTRACTOR
            feats = torch.flatten(feats, 1) # Output: [Batch, 512]

        X_list.append(feats.cpu())
        y_list.append(yb)

    X = torch.cat(X_list).numpy()
    y = torch.cat(y_list).numpy()
    return X, y

OUT_DIR = "results/features"
FEATURE_FILE = os.path.join(OUT_DIR, "gtsrb_resnet18_feats.npz")

def main():


    root_dir = "data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if features already exist
    if os.path.exists(FEATURE_FILE):
        print(f"Features already exist at {FEATURE_FILE}, loading...")
        data = np.load(FEATURE_FILE)
        X_tr = data["X_tr"]
        y_tr = data["y_tr"]
        classes = data["classes"]
        print("Features loaded successfully.")
        return


    feature_extractor = get_feature_extractor(device)

    #Configuration to process train and test
    datasets_config = [
        ("Train", "data/Train.csv"), 
        ("Test", "data/Test.csv")
    ]

    results = {}

    for name, csv_path in datasets_config:
        print(f"\n--- Feature Extraction for {name} Set ---")
        
        if not os.path.exists(csv_path):
            print(f"Errore: Non trovo il file {csv_path}")
            return

        df = load_gtsrb_dataframe(csv_path)
        
        # Dataloader creation
        print("Creating DataLoader and extracting features...")
        loader = DataLoader(
            FunctionalDataset(df, root_dir, transform=get_transforms(), use_roi=True),
            batch_size=256,   
            shuffle=False,    
            num_workers=4,    
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Extraction
        X, y = extract_embeddings(loader, feature_extractor, device)
        
        # Saving on RAM
        results[f"X_{name.lower()}"] = X  # es. X_train
        results[f"y_{name.lower()}"] = y
        print(f"Completato {name}: Estratti {X.shape[0]} campioni.")

    if 'X_train' not in results or 'X_test' not in results:
        print("Errore: Dati mancanti, impossibile salvare.")
        return

    #Extract all classes name
    classes = np.unique(results['y_train']).astype(str)

    X_tr_full = results['X_train']
    y_tr_full = results['y_train']


    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, 
        y_tr_full, 
        test_size=0.20, 
        random_state=42, 
        stratify=y_tr_full
    )

    #Saving on DISC
    os.makedirs(OUT_DIR, exist_ok=True)



    np.savez_compressed(
        FEATURE_FILE, 
        X_tr=X_train, 
        y_tr=y_train, 
        X_val=X_val,
        y_val=y_val,
        X_te=results['X_test'], 
        y_te=results['y_test'], 
        classes=classes
    )
    print(f"Feature extraction completed and saved to {FEATURE_FILE}")

if __name__ == "__main__":
    main()
