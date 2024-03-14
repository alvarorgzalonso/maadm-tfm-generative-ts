import torch
import pytorch_lightning as pl
import torchmetrics

from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from typing import Sequence


class ClassificationModule(pl.LightningModule):
    """
    LightningModule for classification tasks.

    Args:
        model: The classification model.
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
        negative_ratio,
    ):
        super().__init__()
        self.optimizer_config = {
            **self.get_default_optimizer_config(),
            **optimizer_config,
        }
        self.model = model
        self.f1_score = torchmetrics.F1Score(task="binary")

        # store the parameters for the cross entropy loss
        # this is used to correct possible imbalances in the dataset
        self.pos_weight = torch.tensor(negative_ratio, requires_grad=False)
        self.weight = torch.tensor(2.0 / (1. + negative_ratio), requires_grad=False)

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
        probs = torch.sigmoid(logits)
        f1_score = self.f1_score(probs, labels)
        self.log_dict(
            {
                "train_loss": loss,
                "train_f1_score": f1_score,
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
        probs = torch.sigmoid(logits)
        f1_score = self.f1_score(probs, labels)
        self.log_dict(
            {
                "val_loss": loss,
                "val_f1_score": f1_score,
            },
            on_epoch=True,
            on_step=False,
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
        labels = batch["labels"].float().view(-1, 1)

        logits = self.model.forward(batch)  # (BATCH_SIZE, 1)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight, weight=self.weight,
        )
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
        Configures and returns a list of callbacks for the classification trainer.
        In this case:
        * ModelCheckpoint: saves the model with the best validation f1 score.
        * EarlyStopping: stops training if the validation f1 score does not improve for 10 epochs.
        * LearningRateMonitor: logs the learning rate at each step.

        Returns:
            A list of callbacks for the classification trainer.
        """
        return super().configure_callbacks() + [
            ModelCheckpoint(
                filename="{epoch}-{val_f1_score:.2f}",
                monitor="val_f1_score",
                mode="max",
            ),
            EarlyStopping(
                monitor="val_f1_score",
                patience=5,
                mode="max",
            ),
            LearningRateMonitor(logging_interval="step"),
        ]