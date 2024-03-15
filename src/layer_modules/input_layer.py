import torch

from torch import nn

class ModelWithInputLayer(nn.Module):
    def __init__(self, model, input_dim, ff_dim, dropout_p=0.1):
        super(ModelWithInputLayer, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim),
        )
        self.model = model
        self.output_dim = model.output_dim

    def forward(
        self, 
        input,  # (BATCH_SIZE, MAX_LEN)
    ):
        x = self.ff(input)
        return  self.model(x)  # (BATCH_SIZE, OUT_DIM)