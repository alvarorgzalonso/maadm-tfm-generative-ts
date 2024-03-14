import torch
import pytorch_lightning as pl
import torchmetrics

from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from typing import Sequence

class GanModule(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        optimizer_config=None,
        negative_ratio=0.5,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_config = optimizer_config if optimizer_config is not None else self.get_default_optimizer_config()

        self.adversarial_loss = torch.nn.BCELoss()

    @staticmethod
    def get_default_optimizer_config() -> dict:
        return {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
        }

    def training_step(self, batch):
        # Entrenamiento del Discriminador
        real_x = batch
        batch_size = real_x.size(0)

        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # Generamos un conjunto de imágenes falsas
        z = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        fake_x = self.generator(z)

        # Entrenamos el discriminador
        real_loss = self.adversarial_loss(self.discriminator(real_x), valid)
        fake_loss = self.adversarial_loss(self.discriminator(fake_x.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # Log para el discriminador
        self.log("loss/discriminator", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Entrenamiento del Generador
        g_loss = self.adversarial_loss(self.discriminator(fake_x), valid)

        # Log para el generador
        self.log("loss/generator", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": d_loss, "g_loss": g_loss}
