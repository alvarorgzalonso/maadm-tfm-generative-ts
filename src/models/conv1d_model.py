import torch

from torch import nn
from typing import Literal
from models.base_model import BaseModel
from layer_modules.inception_module import InceptionModule

Conv1dModelLayer = Literal[
    "conv1d",
    "relu",
    "max_pool",
    "mean_pool",
    "dropout",
    "batch_norm",
    "ff",
    "residual_block",
]


class FeedForwardLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ff_dim: int,
    ):
        super(FeedForwardLayer, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(in_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, out_dim),
        )
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x,  # (...BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    ):
        #x = x.transpose(-1, -2)  # (...BATCH_SIZE, SEQ_LEN, OUT_CHANNELS)
        x = self.ff(x)
        x = self.layer_norm(x)
        #x = x.transpose(-1, -2)  # (...BATCH_SIZE, OUT_CHANNELS, SEQ_LEN)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        layer_params: list[tuple[Conv1dModelLayer, dict]],
    ):
        super(ResidualBlock, self).__init__()

        self.submodel = nn.Sequential(
            *[Conv1dModelLayerBuilder.build(name, params) for name, params in layer_params]
        )

    def forward(self, x):
        return x + self.submodel(x)


class Conv1dModelLayerBuilder:
    @classmethod
    def build(cls, name: Conv1dModelLayer, params: dict):
        match name:
            case "conv1d":
                return nn.Conv1d(**params)
            case "max_pool":
                return nn.MaxPool1d(**params)
            case "mean_pool":
                return nn.AvgPool1d(**params)
            case "relu":
                return nn.ReLU()
            case "dropout":
                return nn.Dropout(**params)
            case "batch_norm":
                return nn.BatchNorm1d(**params)
            case "ff":
                return FeedForwardLayer(**params)
            case "residual_block":
                return ResidualBlock(**params)
            case "inception_module":
                return InceptionModule(**params)
            case _:
                raise ValueError(f"Invalid Conv1dModelLayer name: {name}")


class Conv1dModel(nn.Module):
    def __init__(
        self,
        output_dim: int,
        layer_params: list[tuple[Conv1dModelLayer, dict]],
    ):
        super(Conv1dModel, self).__init__()

        self.layers = nn.ModuleList(
            Conv1dModelLayerBuilder.build(name, params) for name, params in layer_params
        )

        # calc output dim
        self.output_dim = output_dim

    def _get_out_dim(self):
        return self.output_dim

    def forward(
        self,
        input,  # (...BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    ):
        i = 0
        x = input
        #x = x.transpose(-1, -2)  # (...BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
        for layer in self.layers:
            x = layer(x)
            i += 1
        x = torch.mean(x, dim=-1)  # (...BATCH_SIZE, OUTPUT_DIM)
        return x