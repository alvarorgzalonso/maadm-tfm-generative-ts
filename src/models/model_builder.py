from typing import Literal
from models.conv1d_model import Conv1dModel

Model = Literal["conv1d"]


class ModelBuilder:
    """A class responsible for building different models based on the given name and parameters."""

    @classmethod
    def build(cls, name: Model, params: dict):
        """
        Build a model based on the given name and parameters.

        Args:
            name (Model): The name of the model to build.
            params (dict): The parameters for building the model.

        Returns:
            The built model.

        Raises:
            ValueError: If an invalid model name is provided.
        """
        if "conv1d" in name:
            return cls._build_conv1d_model(params)
        else:
            raise ValueError(f"Invalid model name: {name}")
    
    @classmethod
    def _build_conv1d_model(cls, params: dict):
        """
        Build a Conv1D model based on the given parameters.

        Args:
            params (dict): The parameters for building the Conv1D model.
            num_embeddings (int): The number of embeddings.
            padding_idx (int): The padding index.

        Returns:
            The built Conv1D model.
        """
        return Conv1dModel(**params)
    