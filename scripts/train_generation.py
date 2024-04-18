import os
import argparse
import json
import sys
import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
from torch import nn

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_path, "..", "src"))
sys.path.append(os.path.join(dir_path, "..", "utils"))
from utils import utils_functions as uf


from data_loaders.sine import SineDataModule
from data_loaders.melbourne_pedestrian import MelbounePedestrianDataModule
from models.model_builder import ModelBuilder
from models.inceptiontime import InceptionTime
from trainers.generation_trainer import TSGenerationModule
from layer_modules.input_layer import ModelWithInputLayer



def get_data_module(config: dict):
    """
    Get the data module
    Args:
        config (dict): The configuration dictionary containing various parameters.

    Returns:
        DataModule: The data module.
    """
    if "melbourne_pedestrian" in config["dataset_name"]:
        return MelbounePedestrianDataModule(config["dataset_config"], loader_config=config["loader_config"])
    else:
        raise ValueError("Invalid data name")
    

def run(config: dict, data_module, initial_generator: nn.Module=None):
    """
    Train the model
    Args:
        config (dict): The configuration dictionary containing various parameters.
        initial_generator (Model): The initial generator to be finetuned.
        initial_discriminator (Model): The initial discriminator to be finetuned.

    Returns:
        Model: The finetuned model.
    """
    ### Set up configuration
    model_name = config["model_configs"]["model_name"]
    out_dir = os.path.join('out', config["model_configs"]["out_dir"])

    ### Build model
    if initial_generator is None:
        generator = uf.load_generator(config["model_configs"], data_module)
    else:
        generator = initial_generator

    logger_tb = pl_loggers.TensorBoardLogger(out_dir, name='')
    logger_csv = pl_loggers.CSVLogger(out_dir, name='')
    print(f"Saving to:\nlogger_tb.log_dir: {logger_tb.log_dir}\nlogger_csv.log_dir: {logger_csv.log_dir}")
    
    if not os.path.exists(logger_tb.log_dir): os.makedirs(logger_tb.log_dir)
    
    with open(os.path.join(logger_tb.log_dir, f"{config['model_configs']['out_dir']}.json"), "w") as file:
        config["model"] = str(generator)
        json.dump(config, file, indent=4)


    optimizer_params = config["model_configs"]["optimizer_params"]
    generation_module = TSGenerationModule(
        model=generator,
        optimizer_config=optimizer_params,
        num_classes=data_module.num_classes,
        noise_dim=config["model_configs"]["noise_dim"],
        l2_lambda=0.5,
        l1_lambda=0.5,
        logs_dir=logger_tb.log_dir,
        ckpts_dir=os.path.join(logger_tb.log_dir, "ckpts"),
    )

    trainer_args = config["trainer_args"]
    trainer_args["logger"] = [logger_tb, logger_csv]
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(generation_module, data_module)



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
            "This tool trains a time series generator."
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
        "--out-dir",
        type=str,
        help="Name of the checkpoint to save the model to.",
    )

    args = parser.parse_args()
    out_dir = os.path.join('out', args.out_dir.replace(".ckpt", ""))
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    config_file_model = os.path.join('configs', 'models', args.model_config)
    config_file_data = os.path.join('configs', 'data', args.dataset_config)
    
    if not os.path.exists(config_file_model):
        raise ValueError(f"Please provide a path to the model configuration file.\n{config_file_model} not found.")
    if not os.path.exists(config_file_data):
        raise ValueError(f"Please provide a path to the dataset configuration file.\n{config_file_data} not found.")

    data_module = get_data_module(parse_config(config_file_data))

    configs = {}
    configs['model_configs'] = parse_config(config_file_model)
    configs['model_configs']['out_dir'] = args.out_dir.replace(".ckpt", "")    
    configs["trainer_args"] = {
            "max_steps": 20000,
            "enable_checkpointing": True,
            "default_root_dir": f"out",
            "accelerator": "auto",#"cuda",
    }

    generator = None
        
    run(configs, data_module, generator)
    
    

    

    
