import numpy as np
from torch.utils.data import Dataset
import torch


class TokenDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path, mmap_mode='r')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # gets tokens, then splits into train test pairs
        tokens = torch.tensor(self.data[idx], dtype=torch.long)
        return tokens[:-1], tokens[1:]
