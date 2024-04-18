import torch
import os
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from typing import Sequence

from callbacks.metrics_logger import MetricsLogger

import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
project_src_path = os.path.join(dir_path, "..", "src")
sys.path.append(project_src_path)

from utils import utils_functions as uf

class GANModule(pl.LightningModule):
    """
    LightningModule for GAN tasks.

    Args:
        generator: The generator model.
        discriminator: The discriminator model.
        optimizer_config: Configuration for the optimizer.
        logs_dir: The directory to save the logs.
        ckpts_dir: The directory to save the checkpoints.
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
        generator,
        discriminator,
        optimizer_config,
        num_classes,
        noise_dim=100,
        gan_loss_weight=0.5,
        l2_lambda=0.4,
        l1_lambda=0.1,
        logs_dir: str = os.path.join("out", "classificator"),
        ckpts_dir: str = os.path.join("out", "classificator_ckpts"),
    ):
        super().__init__()
        self.optimizer_config = {
            **self.get_default_optimizer_config(),
            **optimizer_config,
        }
        self.logs_dir = logs_dir
        if not os.path.exists(self.logs_dir): os.makedirs(self.logs_dir)
        self.ckpts_dir = ckpts_dir
        if not os.path.exists(self.ckpts_dir): os.makedirs(self.ckpts_dir)
        
        self.automatic_optimization = False

        self.generator = generator
        self.discriminator = discriminator
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.GAN_LOSS_WEIGHT = gan_loss_weight
        self.L2_LAMBDA = l2_lambda
        self.L1_LAMBDA = l1_lambda
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l2 = nn.MSELoss()

    

    def generator_loss(self, disc_generated_output, gen_output, target):
        """
        This function calculates the total loss for the generator.

        Parameters:
        disc_generated_output (Tensor): The discriminator's prediction on the generated (fake) images
        gen_output (Tensor): The generated (fake) images
        target (Tensor): The real images

        Returns:
        total_gen_loss (Tensor): The total loss for the generator
        gan_loss (Tensor): The GAN loss for the generator
        l2_loss (Tensor): The L2 loss for the generator
        """
        
        total_gen_loss = 0  # Initialize the total loss for the generator
        gan_loss = 0  # Initialize the GAN loss for the generator
        l2_loss = 0  # Initialize the L2 loss for the generator

        gan_loss = F.binary_cross_entropy_with_logits(torch.ones_like(disc_generated_output), disc_generated_output)
        
        l1_loss = self.criterion_l1(gen_output, target)
        l2_loss = self.criterion_l2(gen_output, target)
        
        # Calculate the total loss for the generator by adding the GAN loss (multiplied by its corresponding weight) and the L2 loss (multiplied by the regularization parameter)
        total_gen_loss = self.GAN_LOSS_WEIGHT * gan_loss + self.L2_LAMBDA * l2_loss + self.L1_LAMBDA * l1_loss

        # Return the total loss, GAN loss, and L2 loss for the generator
        return total_gen_loss.clone().detach().requires_grad_(True), gan_loss, l2_loss, l1_loss
    

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """
        This function calculates the total loss for the discriminator.

        Parameters:
        disc_real_output (Tensor): The discriminator's prediction on the real images
        disc_generated_output (Tensor): The discriminator's prediction on the generated (fake) images

        Returns:
        total_disc_loss (Tensor): The total loss for the discriminator
        """
        
        total_disc_loss = 0  # Initialize the total loss for the discriminator
        
        # Calculate the cross entropy loss for the real images
        real_loss = F.binary_cross_entropy_with_logits(torch.ones_like(disc_real_output), disc_real_output)

        # Calculate the cross entropy loss for the generated (synthetic) images
        fake_loss = F.binary_cross_entropy_with_logits(torch.zeros_like(disc_generated_output), disc_generated_output)
        
        # Calculate the total loss for the discriminator by adding the losses for the real and generated images
        total_disc_loss = real_loss + fake_loss

        # Return the total loss for the discriminator
        return total_disc_loss.clone().detach().requires_grad_(True)
    

    def training_step(self, batch):
        """
        Perform a single step of the training process.

        Args:
            batch (dict): A dictionary containing the input batch data.

        Returns:
            tuple: A tuple containing the loss, logits, and labels.
        """
        torch.autograd.set_detect_anomaly(True)
        generator_optimizer, discriminator_optimizer = self.optimizers()

        labels = batch["label"].float().view(-1, self.num_classes)  # (BATCH_SIZE, num_classes)
        target = batch["input"].float().view(-1, batch["input"].shape[1], batch["input"].shape[2])  # (B, C, T)

        noise = torch.randn(labels.shape[0], self.noise_dim)
        noise = noise.type_as(batch["input"])
        noise = torch.cat((noise, labels), dim=1)
        noise = noise.view(-1, batch["input"].shape[1], self.noise_dim + self.num_classes) # (B, C, T)

        labels = labels.view(-1, 1, self.num_classes)
        disc_real_input = torch.cat((target, labels), dim=-1)

        ## Generator training ##
        self.toggle_optimizer(generator_optimizer)
        self.toggle_optimizer(discriminator_optimizer)

        logits = self.generator.forward(noise).view(-1, 1, batch["input"].shape[2])  # (B, C, T)
        fake_ts = torch.tanh(logits)
        fake_disc_input = torch.cat((fake_ts, labels), dim=-1)
        disc_generated_output = self.discriminator(fake_disc_input)

        gen_loss, gan_loss, l2_loss, l1_loss = self.generator_loss(disc_generated_output, fake_ts, target)
        self.manual_backward(gen_loss, retain_graph=True)
        generator_optimizer.step()
        generator_optimizer.zero_grad()
        self.untoggle_optimizer(generator_optimizer)


        ## Discriminator training ##
        disc_real_output = self.discriminator(disc_real_input)
        disc_real_output = disc_real_output.view(-1, 1)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
        self.log_dict(
            {
                "disc_loss": disc_loss,
                "gen_loss": gen_loss,
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=labels.nelement(),
        )
        self.manual_backward(disc_loss)
        discriminator_optimizer.step()
        discriminator_optimizer.zero_grad()
        self.untoggle_optimizer(discriminator_optimizer)

        #return gen_loss, disc_loss, gan_loss, l2_loss, l1_loss
    
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the training process.

        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler.
        """
        generator_optimizer = torch.optim.AdamW(
            params=self.generator.parameters(), **self.optimizer_config
        )
        generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer, mode="min", factor=0.75, patience=3
        )
        discriminator_optimizer = torch.optim.AdamW(
            params=self.discriminator.parameters(), **self.optimizer_config
        )
        discriminator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            discriminator_optimizer, mode="min", factor=0.75, patience=3
        )
        gen_opt_dict = {
            "optimizer": generator_optimizer,
            "lr_scheduler": {
                "scheduler": generator_scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
            }
        }
        disc_opt_dict = {
            "optimizer": discriminator_optimizer,
            "lr_scheduler": {
                "scheduler": discriminator_scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
            }
        }

        return [gen_opt_dict, disc_opt_dict]

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """
        Configures and returns a list of callbacks for the classification trainer.
        In this case:
        * ModelCheckpoint: saves the model with the best validation f1 score.
        * EarlyStopping: stops training if the validation f1 score does not improve for 10 epochs.
        * LearningRateMonitor: logs the learning rate at each step.
        * MetricsLogger: logs the training metrics and classification report at the end of training.

        Returns:
            A list of callbacks for the classification trainer.
        """
        return super().configure_callbacks() + [
            ModelCheckpoint(
                dirpath=self.ckpts_dir,
                filename="{epoch}-{disc_loss:.2f}",
                monitor="disc_loss",
                mode="min",
            ),
            EarlyStopping(
                monitor="disc_loss",
                patience=2,
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
            #MetricsLogger(report_file_path=os.path.join(self.logs_dir, "classification_report.txt")),
        ]