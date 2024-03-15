import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class BotnetDataset(Dataset):
    """
    https://github.com/koumajos/classificationbasedonsfts dataset
    """

    def __init__(self, path_csv):
        """
        Initializes a new instance of the SineDataset class.
        """
        self.data = pd.read_csv(path_csv)
        self.data = self.data.dropna()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)

