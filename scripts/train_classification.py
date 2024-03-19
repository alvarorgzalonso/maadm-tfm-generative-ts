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
from trainers.classification_trainer import ClassificationModule
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
    

def run(config: dict, data_module, initial_classifier: nn.Module=None):
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
    out_dir = os.path.join('out', model_name)

    ### Build model
    if initial_classifier is None:
        input_layer_params = config["model_configs"]["model_params"]["input_layer_params"]
        conv_model_params = config["model_configs"]["model_params"]["conv1d_layers_params"]
        classification_head_params = config["model_configs"]["model_params"]["classification_head_params"]
        
        classification_head_params["num_classes"] = data_module.num_classes
        conv_model_params["layer_params"][0][1]["in_channels"] = data_module.n_channels

        name = f"conv1d_{model_name}"
        print(f"Building classifier {name}...")
        
        input_layer_params["input_dim"] = data_module.n_timepoints

        model = ModelBuilder.build(name, conv_model_params)
        model = ModelWithInputLayer(model, **input_layer_params)
        classifier = ModelWithClassificationHead(
            model,
            model.output_dim,
            **classification_head_params,
        )
    else:
        print("Using initial generator")
        classifier = initial_classifier

    logger = pl_loggers.TensorBoardLogger(out_dir, name='')

    if not os.path.exists(logger.log_dir): os.makedirs(logger.log_dir)
    
    with open(os.path.join(logger.log_dir, f"{config['model_configs']['ckpt_name']}.json"), "w") as file:
        config["model"] = str(classifier)
        json.dump(config, file, indent=4)


    optimizer_params = config["model_configs"]["optimizer_params"]
    classification_module = ClassificationModule(
        model=classifier,
        optimizer_config=optimizer_params,
        num_classes=data_module.num_classes,
        negative_ratio=(1. / data_module.get_positive_ratio()),
        logs_dir=logger.log_dir,
        ckpts_dir=os.path.join(logger.log_dir, "ckpts"),
    )

    trainer_args = config["trainer_args"]
    trainer_args["logger"] = logger
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(classification_module, data_module)



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
    out_dir = os.path.join('out', args.ckpt_name.replace(".ckpt", ""))
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
    configs['model_configs']['ckpt_name'] = args.ckpt_name.replace(".ckpt", "")    
    configs["trainer_args"] = {
            "max_steps": 20000,
            "enable_checkpointing": True,
            "default_root_dir": f"out",
            "accelerator": "auto",#"cuda",
    }
    
    classifier = InceptionTime(n_classes = data_module.num_classes, in_channels = data_module.n_channels, kszs=[10, 20, 40])
    #classifier = None
    run(configs, data_module, classifier)
    
    

    

    
