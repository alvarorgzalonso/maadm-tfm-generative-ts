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
from trainers.classification_trainer import ClassificationModule
from layer_modules.classification_head import ModelWithClassificationHead
from layer_modules.input_layer import ModelWithInputLayer


def run(config: dict, initial_classifier: nn.Module=None):
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
        data_module = SineDataModule(config["data_configs"]["dataset_config"], loader_config=config["data_configs"]["loader_config"])
    else:
        raise ValueError("Invalid data name")
    
    model_name = config["model_configs"]["model_name"]
    input_layer_params = config["model_configs"]["model_params"]["input_layer_params"]
    conv_model_params = config["model_configs"]["model_params"]["conv1d_layers_params"]
    classification_head_params = config["model_configs"]["model_params"]["classification_head_params"]
    
    if initial_classifier is None:
        name = f"conv1d_{model_name}"
        print(f"Building classifier {name}...")
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
    
    #Redirect print to json file
    with open(f"out/metadata/{config['model_configs']['ckpt_name']}.json", "w") as file:
        config["model"] = str(classifier)
        json.dump(config, file, indent=4)

    trainer_args = config["trainer_args"]

    optimizer_params = config["model_configs"]["optimizer_params"]
    classification_module = ClassificationModule(
        model=classifier,
        optimizer_config=optimizer_params,
        negative_ratio=(1. / data_module.get_positive_ratio()),
    )
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
    
    configs["trainer_args"] = {
            "max_steps": 20000,
            "enable_checkpointing": True,
            "default_root_dir": f"out/{checkpoint_path}",
            "accelerator": "auto",#"cuda",
    }
        
    run(configs)
    
    

    

    
