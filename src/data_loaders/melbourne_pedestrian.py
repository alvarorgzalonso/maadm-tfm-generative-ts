import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from fastai.data.external import untar_data
import pprint

from sklearn.preprocessing import OneHotEncoder

class MelbounePedestrianDataset(Dataset):
    def __init__(self, split="train", vocab=None):
        self.vocab = vocab

        path = untar_data('https://timeseriesclassification.com/aeon-toolkit/MelbournePedestrian.zip')
        train_data = os.path.join(str(path), 'MelbournePedestrian_TRAIN.txt')
        test_data = os.path.join(str(path), 'MelbournePedestrian_TEST.txt')
        if split == "train":
            data_path = train_data
        elif split == "test":
            data_path = test_data

        train_df = pd.read_csv(data_path, sep='  ', header=None)
        train_df[0] = train_df[0] - 1

        n_samples = train_df.shape[0]
        self.n_channels = 1
        self.n_timepoints = train_df.shape[1] - 1

        self.uniqiue_labels = train_df[0].unique().tolist()
        
        self.num_classes = len(self.uniqiue_labels)

        self.y = torch.tensor(train_df[0].values)
        self.X = torch.tensor(train_df.iloc[:, 1:].values, dtype=torch.float32).view(n_samples, self.n_channels, self.n_timepoints)
        
        self.scaler_min_max()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.vocab:
            return self.X[idx], self.vocab[self.y[idx]]
        else:
            return self.X[idx], self.y[idx]
    
    def scaler_min_max(self):
        # save the min and max of the training set
        self.min = self.X.min()
        self.max = self.X.max()
        # scale the training set
        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
        self.X = self.X * 2 - 1

class MelbounePedestriancollatorFn:
    def __init__(self, vocab=None):
        self.vocab = vocab
        

    def __call__(self, batch):
        X, y = list(zip(*batch))
        X = torch.stack(X)
        y = torch.tensor(y)
        if self.vocab:
            y = torch.tensor([self.vocab[i.item()] for i in y])
        return {"input": X, "label": y}
    
    
class MelbounePedestrianDataModule(pl.LightningDataModule):

    @classmethod
    def get_default_loader_config(cls):
        return {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
        }

    @classmethod
    def get_default_dataset_config(cls):
        return {}
    
    @classmethod
    def get_default_collator_config(cls):
        return {
            "vocab": None,
        }
    
    @classmethod
    def from_joint_config(cls, config):
        """
        Create an instance of MelbounePedestrianDataModule from a single dictionary of configuration.
        This dictionary is split into two parts: one for the collator function and one
        for the data loader.

        Args:
            config (dict): The joint configuration.

        Returns:
            MelbounePedestrianDataModule: The data module instance.
        """
        collator_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_collator_config()
        }
        loader_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_loader_config()
        }
        dataset_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_dataset_config()
        }
        return cls(dataset_config, loader_config=loader_config, collator_config=collator_config)
    
    def __init__(self, dataset_config: dict, loader_config: dict = {}, collator_config: dict = {}):
        super().__init__()

        # build loader config
        self.dataset_config = {
            **self.get_default_dataset_config(),
            **dataset_config,
        }

        # load datasets
        self.train_dataset = MelbounePedestrianDataset(**{**self.dataset_config, "split": "train"})
        self.val_dataset = MelbounePedestrianDataset(**{**self.dataset_config, "split": "test"})

        self.num_classes = self.train_dataset.num_classes
        self.n_channels = self.train_dataset.n_channels
        self.n_timepoints = self.train_dataset.n_timepoints

        # build collator_fn
        self.collator_config = {
            **self.get_default_collator_config(),
            **collator_config,
        }
        
        self.one_hot_encoder = OneHotEncoder(categories=[self.train_dataset.uniqiue_labels], sparse_output=False)
        one_hot_labels = self.one_hot_encoder.fit_transform(np.array(self.train_dataset.uniqiue_labels).reshape(-1, 1))

        collator_config["vocab"] = {v: one_hot_labels[i] for i, v in enumerate(self.train_dataset.uniqiue_labels)}
        #collator_config["vocab"] = {"".join(str(one_hot_labels[i])): v for i, v in enumerate(self.train_dataset.uniqiue_labels)}
        
        # build collator_fn
        self.loader_config = {
            **self.get_default_loader_config(),
            **loader_config,
            "collate_fn": MelbounePedestriancollatorFn(**collator_config),
        }
        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.loader_config,
            shuffle=False,
        )
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            **self.loader_config,
            shuffle=False,
        )
    
    def get_positive_ratio(self):
        return 1/self.num_classes