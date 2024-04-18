import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import torch

import os
import sys
from sktime.distances import distance_factory

dir_path = os.path.dirname(os.path.abspath(__file__))
project_src_path = os.path.join(dir_path, "..", "src")
sys.path.append(project_src_path)

from models.model_builder import ModelBuilder
from layer_modules.input_layer import ModelWithInputLayer
from layer_modules.classification_head import ModelWithClassificationHead

def parse_config(config_file_name):
    """
    Load json config
    """
    try:
        with open(config_file_name, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{config_file_name}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{config_file_name}' is not a valid JSON file.")
        return

def load_generator(config, data_module):
    """
    Load a generator from a configuration.
    """
    generator_name = config["model_name"]

    input_layer_params = config["model_params"]["input_layer_params"]
    conv_model_params = config["model_params"]["conv1d_layers_params"]

    name = f"conv1d_{generator_name}"
    print(f"Building generator {name}...")

    input_layer_params["input_dim"] = config["noise_dim"] + data_module.num_classes

    model = ModelBuilder.build(name, conv_model_params)
    generator = ModelWithInputLayer(model, **input_layer_params)
    return generator

def load_discriminator(config, data_module):
    """
    Load a discriminator from a configuration.
    """
    discriminator_name = config["model_name"]

    input_layer_params = config["model_params"]["input_layer_params"]
    conv_model_params = config["model_params"]["conv1d_layers_params"]
    #conv_model_params["layer_params"][0][1]["in_channels"] = data_module.n_channels

    name = f"conv1d_{discriminator_name}"
    print(f"Building discriminator {name}...")

    input_layer_params["input_dim"] = data_module.n_timepoints

    model = ModelBuilder.build(name, conv_model_params)
    model = ModelWithInputLayer(model, **input_layer_params)
    try:
        classification_head_params = config["model_params"]["classification_head_params"]
        classification_head_params["num_classes"] = data_module.num_classes
        discriminator = ModelWithClassificationHead(
            model,
            model.output_dim,
            **classification_head_params,
        )
    except Exception as e:
        print(f"Sin cabeza de clasificación({e})")
        discriminator = model
    return discriminator

def load_state_dict(model, ckpt_path):
    """
    Load a state dict into a model.
    """
    checkpoint_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    state_dict = {key.replace('model.model', 'model').replace('model.ff', 'ff'): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model, state_dict

def one_batch_gen(it, generator, NUM_CLASSES, NOISE_DIM, L2_LAMBDA, L1_LAMBDA):

    batch = next(it)
    labels = batch["label"].float().view(-1, NUM_CLASSES)
    noise = torch.randn(labels.shape[0], NOISE_DIM)
    noise = noise.type_as(batch["input"])
    noise = torch.cat((noise, labels), dim=1)
    noise = noise.view(-1, 1, NOISE_DIM + NUM_CLASSES) # (B, C, T)
    
    generated = generator(noise)
    
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    target = batch["input"].float().view(-1,batch["input"].shape[1], batch["input"].shape[2])  # (B, C, T)
    total_gen_loss = L2_LAMBDA * l2_loss(generated, target) + L1_LAMBDA * l1_loss(generated, target)
    print(f"total_gen_loss: {total_gen_loss}")
    print(f"l1_loss: {l1_loss(generated, target)}")
    print(f"l2_loss: {l2_loss(generated, target)}")
    return generated, noise, target, labels, total_gen_loss


def find_lr(model, data_module, trainer_args, save_path=None, min_lr=1e-8, max_lr=10.0):
    trainer = pl.Trainer(**trainer_args)
    tunner = pl.tuner.Tuner(trainer)
    results = tunner.lr_find(
        model,
        attr_name="lr",
        train_dataloaders=data_module, 
        min_lr=min_lr,
        max_lr=max_lr,
    )
    # Plot with
    fig = results.plot(suggest=True)
    # save the figure
    if save_path: fig.savefig(save_path)
    fig.show()
    lr = results.suggestion()
    print(f"Learning rate suggestion: {lr}")
    return lr


def get_msm_fn(target, generated, dataset):
    reescaled_target = dataset.inverse_scaler_min_max(target).detach().numpy().reshape(-1)
    reescaled_generated = dataset.inverse_scaler_min_max(generated).detach().numpy().reshape(-1)
    return distance_factory(reescaled_generated, reescaled_target, "msm")