import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, path, split='train'):
        """
        split: 'train' carica X_tr/y_tr, 'test' carica X_te/y_te
        """
        data = np.load(path)
        if split == 'train':
            self.X = torch.from_numpy(data['X_tr']).float()
            self.y = torch.from_numpy(data['y_tr']).long()
        else:
            self.X = torch.from_numpy(data['X_te']).float()
            self.y = torch.from_numpy(data['y_te']).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]