import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from typing import Tuple


class TwoMoonsDataset:
    def __init__(self, n_samples=10000, noise=0.08, random_state=0, normalize=True):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.normalize = normalize
        
        self.data = None
        self.mean = None
        self.std = None
        
        self._generate_data()
    
    def _generate_data(self):
        X, _ = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        
        X = X.astype(np.float32)
        
        if self.normalize:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            X = (X - self.mean) / (self.std + 1e-6)
        
        self.data = X
    
    def denormalize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return X
        return (X * (self.std + 1e-6)) + self.mean
    
    def create_dataloader(self, batch_size=256, shuffle=True, drop_last=True) -> DataLoader:
        data_tensor = torch.from_numpy(self.data)
        
        dataset = TensorDataset(data_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
