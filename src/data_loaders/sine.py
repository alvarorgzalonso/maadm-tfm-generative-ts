import numpy as np
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class SineDataset(Dataset):
    """
    A custom dataset class for generatin sine function data.

    Args:
        len_dataset (int): the number of samples.
        dim (int): feature dimensions

    Attributes:
        len (int): The number of data files in the directory.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the data item at the specified index.
    """
  
    def __init__(self, len_dataset, dim, seq_len):
        """
        Initializes a new instance of the SineDataset class.
        """
        self.data = list()
        self.len = len_dataset
        self.dim = dim
        self.seq_len = seq_len

        for i in range(len):
            temp = list()
            for k in range(dim):
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)
                temp_data = [np.sin(freq * j + phase) for j in range(self.seq_len)]
                temp.append(temp_data)
            temp = np.transpose(np.asarray(temp))
            temp = (temp + 1)*0.5
            self.data.append(temp)
        
        self.scaler_min_max()


    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of data files in the directory.
        """
        return self.len
    

    def __getitem__(self, index):
        """
        Returns the data item at the specified index.

        Args:
            index (int): The index of the data item to return.

        Returns:
            torch.Tensor: The data item at the specified index.
        """
        return torch.tensor(self.norm_data[index], dtype=torch.float32)
    

    def scaler_min_max(self):
        """
        Scales the data using min-max scaler. Saves the min and max values for each feature.
        """
        self.min_val = np.min(np.min(self.data, axis=0), axis=0)
        self.data = (self.data - self.min_val)
        self.max_val = np.max(np.max(self.data, axis=0), axis=0)
        self.norm_data = self.data / (self.max_val + 1e-6)


class SineCollatorFn:
    """
    A custom collator function for the sine dataset.

    Args:
        dim (int): feature dimensions
        seq_len (int): sequence length

    Methods:
        __call__(batch): Collates the batch of data items.
    """
    def __init__(self,**kwargs):
        """
        Initializes a new instance of the SineCollatorFn class.
        """
        pass


    def __call__(self, batch):
        """
        Collates the batch of data items.

        Args:
            batch (list): The batch of data items to collate.

        Returns:
            torch.Tensor: The collated batch of data items.
        """
        return torch.stack(batch, dim=0)
    

class SineDataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading Sine data.

    Args:
        dataset_config (dict): Configuration for the collator function.
        dataset_config (dict): Configuration for the data loader.

    Attributes:
        dataset_config (dict): Configuration for the data loader.
        train_dataset (PAN23Dataset): The training dataset.
        val_dataset (PAN23Dataset): The validation dataset.
    """

    @classmethod
    def get_default_collator_config(cls):
        return {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
        }

    @classmethod
    def get_default_dataset_config(cls):
        return {
            "len": 1000,
            "dim": 5,
            "seq_len": 100,
        }

    @classmethod
    def from_joint_config(cls, config):
        """
        Create an instance of SineDataModule from a single dictionary of configuration.
        This dictionary is split into two parts: one for the collator function and one
        for the data loader.

        Args:
            config (dict): The joint configuration.

        Returns:
            SineDataModule: The data module instance.
        """
        collator_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_collator_config()
        }
        dataset_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_dataset_config()
        }
        return cls(dataset_config, collator_config)

    def __init__(self, dataset_config: dict, collator_config: dict = {}):
        super().__init__()

        # build collator_fn
        collator_config = {
            **self.get_default_collator_config(),
            **collator_config,
        }
        # build loader config
        self.dataset_config = {
            **self.get_default_dataset_config(),
            **dataset_config,
            "collate_fn": SineCollatorFn(**collator_config),
        }

        # load datasets
        self.train_dataset = SineDataset(**dataset_config)
        self.val_dataset = SineDataset(**dataset_config)

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: A DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.train_dataset,
            **self.dataset_config,
            shuffle=False,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: A DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self.val_dataset,
            **self.dataset_config,
            shuffle=False,
        )

    def get_positive_ratio(self):
        """
        Returns the ratio of positive examples of the training dataset.

        Returns:
            float: The positive ratio of the training dataset.
        """
        return 0.5
    