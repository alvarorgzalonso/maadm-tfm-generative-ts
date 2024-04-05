import os
import argparse
import json
import sys
import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
from torch import nn

dir_path = os.path.dirname(os.path.abspath(__file__))
project_src_path = os.path.join(dir_path, "..", "src")
sys.path.append(project_src_path)


from data_loaders.sine import SineDataModule
from data_loaders.melbourne_pedestrian import MelbounePedestrianDataModule
from models.model_builder import ModelBuilder
from models.inceptiontime import InceptionTime
from trainers.gan_trainer import GANModule
from layer_modules.classification_head import ModelWithClassificationHead
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
    

def run(config: dict, data_module, initial_generator: nn.Module=None, initial_discriminator: nn.Module=None):
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
    out_dir = config["generator_configs"]["out_dir"]

    ### Build generator
    if initial_generator is None:
        generator_name = config["generator_configs"]["model_name"]

        input_layer_params = config["generator_configs"]["model_params"]["input_layer_params"]
        conv_model_params = config["generator_configs"]["model_params"]["conv1d_layers_params"]
        #conv_model_params["layer_params"][0][1]["in_channels"] = data_module.n_channels

        name = f"conv1d_{generator_name}"
        print(f"Building generator {name}...")
        
        input_layer_params["input_dim"] = data_module.num_classes + config["generator_configs"]["noise_dim"]

        model = ModelBuilder.build(name, conv_model_params)
        generator = ModelWithInputLayer(model, **input_layer_params)
        #generator = 
    else:
        print("Using initial generator")
        generator = initial_generator

        
    ### Build discriminator
    if initial_discriminator is None:
        discriminator_name = config["discriminator_configs"]["model_name"]

        input_layer_params = config["discriminator_configs"]["model_params"]["input_layer_params"]
        conv_model_params = config["discriminator_configs"]["model_params"]["conv1d_layers_params"]
        #conv_model_params["layer_params"][0][1]["in_channels"] = data_module.n_channels

        name = f"conv1d_{discriminator_name}"
        print(f"Building discriminator {name}...")
        
        input_layer_params["input_dim"] = data_module.n_timepoints

        model = ModelBuilder.build(name, conv_model_params)
        model = ModelWithInputLayer(model, **input_layer_params)
        try:
            classification_head_params = config["discriminator_configs"]["model_params"]["classification_head_params"]
            classification_head_params["num_classes"] = data_module.num_classes
            discriminator = ModelWithClassificationHead(
                model,
                model.output_dim,
                **classification_head_params,
            )
        except Exception as e:
            print(f"Error: {e}")
            discriminator = model
    else:
        print("Using initial discriminator")
        discriminator = initial_discriminator

    logger_tb = pl_loggers.TensorBoardLogger(out_dir, name='')
    logger_csv = pl_loggers.CSVLogger(out_dir, name='')
    print(f"Saving to:\nlogger_tb.log_dir: {logger_tb.log_dir}\nlogger_csv.log_dir: {logger_csv.log_dir}")
    
    if not os.path.exists(logger_tb.log_dir): os.makedirs(logger_tb.log_dir)
    
    with open(os.path.join(logger_tb.log_dir, f"{os.path.basename(config['generator_configs']['out_dir'])}.json"), "w") as file:
        config["discriminator"] = str(discriminator)
        config["generator"]  = str(generator)
        json.dump(config, file, indent=4)


    optimizer_params = config["generator_configs"]["optimizer_params"]
    gan_module = GANModule(
        generator=generator,
        discriminator=discriminator,
        optimizer_config=optimizer_params,
        num_classes=data_module.num_classes,
        noise_dim=config["generator_configs"]["noise_dim"],
        gan_loss_weight=0.5,
        l2_lambda=0.4,
        l1_lambda=0.1,
        logs_dir=logger_tb.log_dir,
        ckpts_dir=os.path.join(logger_tb.log_dir, "ckpts"),
    )

    trainer_args = config["trainer_args"]
    trainer_args["logger"] = [logger_tb, logger_csv]
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
            "This tool trains a GAN model to generate time series data by taking original as reference."
        ),
    )

    parser.add_argument(
        "--generator-config",
        type=str,
        help="Path to the generator configuration file.",
    )

    parser.add_argument(
        "--discriminator-config",
        type=str,
        help="Path to the discriminator configuration file.",
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
    out_dir = os.path.join('out', args.out_dir)
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    config_file_generator = os.path.join('configs', 'models', args.generator_config)
    config_file_discriminator = os.path.join('configs', 'models', args.discriminator_config)
    config_file_data = os.path.join('configs', 'data', args.dataset_config)
    
    if not os.path.exists(config_file_generator):
        raise ValueError(f"Please provide a path to the generator configuration file.\n{config_file_generator} not found.")
    if not os.path.exists(config_file_discriminator):
        raise ValueError(f"Please provide a path to the discriminator configuration file.\n{config_file_discriminator} not found.")
    if not os.path.exists(config_file_data):
        raise ValueError(f"Please provide a path to the dataset configuration file.\n{config_file_data} not found.")

    data_module = get_data_module(parse_config(config_file_data))

    configs = {}
    configs['generator_configs'] = parse_config(config_file_generator)
    configs['discriminator_configs'] = parse_config(config_file_discriminator)
    configs['generator_configs']['out_dir'] = out_dir
    configs['discriminator_configs']['out_dir'] = out_dir
    configs["trainer_args"] = {
            "max_steps": 20000,
            "enable_checkpointing": True,
            "default_root_dir": f"out",
            "accelerator": "auto",#"cuda",
    }
    
    if "InceptionTime" in configs['discriminator_configs']['model_name']:
        discriminator = InceptionTime(n_classes = data_module.num_classes, in_channels = data_module.n_channels, kszs=[10, 20, 40])
    else:
        discriminator = None
    run(configs, data_module, initial_discriminator=discriminator)