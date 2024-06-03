import torch
from torch.utils.data import Dataset

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # generate a random 4D tensor
        tensor = torch.randn(self.size)
        return tensor