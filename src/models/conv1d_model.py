import torch
from torch import nn

from layer_modules.conv_modules import Conv1dModelLayerBuilder, Conv1dModelLayer



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