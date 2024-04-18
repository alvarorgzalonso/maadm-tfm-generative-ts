import torch
import os
import pytorch_lightning as pl
import torchmetrics
import numpy as np
import torch.nn as nn

from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from typing import Sequence

from callbacks.metrics_logger import MetricsLogger


class TSGenerationModule(pl.LightningModule):
    """
    LightningModule for generation tasks.

    Args:
        model: The generation model.
        optimizer_config: Configuration for the optimizer.
        negative_ratio: Ratio of positive samples in the dataset, to correct imbalance.
    """

    @classmethod
    def get_default_optimizer_config(cls) -> dict:
        """
        Returns the default configuration for the optimizer.

        Returns:
            dict: The default optimizer configuration.
        """
        return {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
        }
    

    def __init__(
        self,
        model,
        optimizer_config,
        num_classes,
        lr=1e-4,
        noise_dim=100,
        l2_lambda=0.5,
        l1_lambda=0.5,
        logs_dir: str = os.path.join("out", "classificator"),
        ckpts_dir: str = os.path.join("out", "classificator_ckpts"),
    ):
        super().__init__()
        self.optimizer_config = {
            **self.get_default_optimizer_config(),
            **optimizer_config,
        }
        self.lr = optimizer_config.get("lr", lr)
        self.learning_rate = self.lr
        self.logs_dir = logs_dir
        if not os.path.exists(self.logs_dir): os.makedirs(self.logs_dir)
        self.ckpts_dir = ckpts_dir
        if not os.path.exists(self.ckpts_dir): os.makedirs(self.ckpts_dir)

        self.noise_dim = noise_dim
        self.L2_LAMBDA = l2_lambda
        self.L1_LAMBDA = l1_lambda

        self.model = model        
        self.num_classes = num_classes
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l2 = nn.MSELoss()

    def loss_fn(self, gen_output, target):
        """
        This function calculates the total loss for the generator.

        Parameters:
        gen_output (Tensor): The generated (fake) images
        target (Tensor): The real images

        Returns:
        total_gen_loss (Tensor): The total loss for the generator
        gan_loss (Tensor): The GAN loss for the generator
        l2_loss (Tensor): The L2 loss for the generator
        """
        
        total_gen_loss = 0  # Initialize the total loss for the generator

        
        self.l1_loss = self.criterion_l1(gen_output, target)
        self.l2_loss = self.criterion_l2(gen_output, target)
        
        # Calculate the total loss for the generator by adding the GAN loss (multiplied by its corresponding weight) and the L2 loss (multiplied by the regularization parameter)
        total_gen_loss = self.L2_LAMBDA * self.l2_loss + self.L1_LAMBDA * self.l1_loss

        # Return the total loss, GAN loss, and L2 loss for the generator
        return total_gen_loss

    def training_step(self, batch):
        """
        Performs a single training step. Logs the training loss and f1 score.

        Args:
            batch (dict): A dictionary containing the input batch data.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the loss, logits, and labels.
        """
        loss, logits, labels = self._step(batch)
        

        self.log_dict(
            {
                "train_loss": loss,
                "l2_loss": self.l2_loss,
                "l1_loss": self.l1_loss,
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=labels.nelement(),
        )
        return {"loss": loss, "logits": logits, "labels": labels}

    def validation_step(self, batch):
        """
        Performs a single validation step. Logs the validation loss and f1 score.

        Args:
            batch (dict): A dictionary containing the input batch data.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the loss, logits, and labels.
        """
        loss, logits, labels = self._step(batch)

        self.log_dict(
            {
                "val_loss": loss,
                "l2_loss": self.l2_loss,
                "l1_loss": self.l1_loss,
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=labels.nelement(),
        )
        return {"loss": loss, "logits": logits, "labels": labels}

    def _step(self, batch):
        """
        Perform a single step of the training process. This method is used by both
        `training_step` and `validation_step` to avoid code duplication.

        Args:
            batch (dict): A dictionary containing the input batch data.

        Returns:
            tuple: A tuple containing the loss, logits, and labels.
        """
        labels = batch["label"].float().view(-1, self.num_classes)  # (BATCH_SIZE, num_classes)
        target = batch["input"].float().view(-1,batch["input"].shape[1], batch["input"].shape[2])  # (B, C, T)
        
        noise = torch.randn(labels.shape[0], self.noise_dim)
        noise = noise.type_as(batch["input"])
        noise = torch.cat((noise, labels), dim=1)
        noise = noise.view(-1, 1, self.noise_dim + self.num_classes)

        logits = self.model.forward(noise).view(-1, 1, batch["input"].shape[2])  # (B, C, T)
        predictions = torch.tanh(logits)
        loss =  self.loss_fn(predictions, target)

        return loss, logits, labels

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the training process.

        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(), **self.optimizer_config
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.75, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
            },
        }

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """
        Configures and returns a list of callbacks for the generation trainer.
        In this case:
        * ModelCheckpoint: saves the model with the best validation f1 score.
        * EarlyStopping: stops training if the validation f1 score does not improve for 10 epochs.
        * LearningRateMonitor: logs the learning rate at each step.
        * MetricsLogger: logs the training metrics and generation report at the end of training.

        Returns:
            A list of callbacks for the generation trainer.
        """
        return super().configure_callbacks() + [
            ModelCheckpoint(
                dirpath=self.ckpts_dir,
                filename="{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                mode="max",
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="max",
            ),
            LearningRateMonitor(logging_interval="step"),
        ]