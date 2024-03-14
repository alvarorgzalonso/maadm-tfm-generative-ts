import os
import argparse
import json
import sys
import pytorch_lightning as pl


from copy import deepcopy
from torch import nn

dir_path = os.path.dirname(os.path.abspath(__file__))
project_src_path = os.path.join(dir_path, "..", "src")
sys.path.append(project_src_path)


from data_loaders.sine import SineDataModule
from models.model_builder import ModelBuilder
from trainers.gan_trainer import GanModule


def run(config: dict, initial_generator: nn.Module=None, initial_discriminator: nn.Module=None):
    """
    Train the model
    Args:
        config (dict): The configuration dictionary containing various parameters.
        initial_generator (Model): The initial generator to be finetuned.
        initial_discriminator (Model): The initial discriminator to be finetuned.

    Returns:
        Model: The finetuned model.
    """
    print("Building data module...")
    if "sine" in config["data_configs"]["dataset_name"]:
        data_module = SineDataModule(config["data_configs"]["dataset_config"], config["data_configs"]["collator_config"])
    else:
        raise ValueError("Invalid data name")
    
    model_name = config["model_configs"]["model_name"]
    
    if initial_generator is None:
        print("Building generator...")
        model_name = config["model_configs"]["model_params"]["generator"]
        generator_name = f"conv1d_{model_name}"
        generator = ModelBuilder.build(f"conv1d_{generator_name}", config["model_configs"]["model_params"]["generator_params"])
    else:
        print("Using initial generator")
        generator = initial_generator

    if initial_discriminator is None:
        print("Building discriminator...")
        model_name = config["model_configs"]["model_params"]["discriminator"]
        discriminator_name = f"conv1d_{model_name}"
        discriminator = ModelBuilder.build(f"conv1d_{discriminator_name}", config["model_configs"]["model_params"]['discriminator_params'])
    else:
        print("Using initial discriminator")
        discriminator = initial_discriminator

    gan_module = GanModule(generator, discriminator, config["model_configs"]["optimizer_config"])

    return None
    trainer_args = config["trainer_args"]
    trainer = pl.Trainer(**trainer_args)

    trainer.fit(gan_module, data_module)



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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer generative model for time series generation.",
        epilog=(
            "This tool trains a GAN or diffusion model to generate time series data by taking original as reference."
        ),
    )

    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to the model configuration file.",
    )

    parser.add_argument(
        "--dataset-config",
        type=str,
        help="Name of the dataset to transform.",
    )

    parser.add_argument(
        "--ckpt-name",
        type=str,
        help="Name of the checkpoint to save the model to.",
    )

    args = parser.parse_args()

    out_metadata_dir = os.path.join('out', 'metadata')
    checkpoint_path = os.path.join('out', 'ckpt')
    if not os.path.exists(out_metadata_dir):
        os.makedirs(out_metadata_dir)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    config_file_model = os.path.join('configs', 'models', args.model_config)
    config_file_data = os.path.join('configs', 'data', args.dataset_config)
    if not os.path.exists(config_file_model):
        raise ValueError(f"Please provide a path to the model configuration file.\n{config_file_model} not found.")
    if not os.path.exists(config_file_data):
        raise ValueError(f"Please provide a path to the dataset configuration file.\n{config_file_data} not found.")
    
    configs = {}
    configs['model_configs'] = parse_config(config_file_model)
    configs['data_configs'] = parse_config(config_file_data)
    configs['model_configs']['ckpt_name'] = args.ckpt_name.replace(".ckpt", "")
    configs['data_configs']['dataset_name'] = args.dataset_config
    
    run(configs)
    
    

    

    
