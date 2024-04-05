import torch

from torch import nn

class ModelWithInputLayer(nn.Module):
    def __init__(self, model, input_dim, ff_dim, output_dim=None, dropout_p=0.1, enable_ff=True):
        super(ModelWithInputLayer, self).__init__()
        if output_dim is not None: self.output_dim = output_dim
        else: self.output_dim = input_dim
        self.model = model
        self.output_dim = model.output_dim
        self.enable_ff = enable_ff
        
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(ff_dim, self.output_dim),
        )

    def forward(
        self, 
        input,  # (BATCH_SIZE, MAX_LEN)
    ):
        if self.enable_ff: x = self.ff(input)
        else: x = input
        
        return  self.model(x)  # (BATCH_SIZE, OUT_DIM)