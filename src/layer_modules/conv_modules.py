import torch.nn as nn
from typing import Literal
import torch


Conv1dModelLayer = Literal[
    "conv1d",
    "relu",
    "max_pool",
    "mean_pool",
    "dropout",
    "batch_norm",
    "ff",
    "residual_block",
    "inception_block",
    "relu",
    "adaptive_avg_pool",
    "flatten",
    "linear",
]

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
            case "inception_block":
                return InceptionBlock(**params)
            case _:
                raise ValueError(f"Invalid Conv1dModelLayer name: {name}")
            
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


class InceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_layer_params: list[tuple[Conv1dModelLayer, dict]],
            nb_filters=32,
            bottleneck_size=32,
            use_bottleneck=True):
        
        super(InceptionBlock, self).__init__()
        
        self.use_bottleneck = use_bottleneck
        if use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
            in_channels = bottleneck_size
        
        self.kernels = nn.Sequential(
            *[Conv1dModelLayerBuilder.build(name, params) for name, params in kernel_layer_params]
        )
                
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm1d(nb_filters * (len(kernel_layer_params) + 1))
        self.act = nn.ReLU()
    
    def forward(self, x):
        if self.use_bottleneck:
            x = self.bottleneck(x)
        
        kernel_out = self.kernels(x)
        pool_out = self.pool_conv(self.max_pool(x))
        out = torch.cat(kernel_out + [pool_out], dim=1)
        
        out = self.bn(out)
        return self.act(out)