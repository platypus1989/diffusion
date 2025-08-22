import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from typing import Tuple


class TwoMoonsDataset:
    def __init__(self, n_samples=10000, noise=0.08, random_state=0, normalize=True, test_ratio=0.2):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.normalize = normalize
        self.test_ratio = test_ratio
        
        self.train_data = None
        self.test_data = None
        self.mean = None
        self.std = None
        
        self._generate_data()
    
    def _generate_data(self):
        X, _ = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        
        X = X.astype(np.float32)
        
        self.train_data, self.test_data = train_test_split(
            X, test_size=self.test_ratio, random_state=self.random_state
        )
        
        if self.normalize:
            # Use only training data for normalization statistics
            self.mean = self.train_data.mean(axis=0)
            self.std = self.train_data.std(axis=0)
            
            self.train_data = (self.train_data - self.mean) / (self.std + 1e-6)
            self.test_data = (self.test_data - self.mean) / (self.std + 1e-6)
        
    def denormalize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return X
        return (X * (self.std + 1e-6)) + self.mean
    
    def create_dataloader(self, batch_size=256, shuffle=True, drop_last=True, split='train') -> DataLoader:
        if split == 'train':
            data_tensor = torch.from_numpy(self.train_data)
        elif split == 'test':
            data_tensor = torch.from_numpy(self.test_data)
        else:
            raise ValueError("split must be 'train' or 'test'")
        
        dataset = TensorDataset(data_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    