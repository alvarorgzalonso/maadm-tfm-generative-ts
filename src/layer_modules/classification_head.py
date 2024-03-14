import torch

from torch import nn

class ModelWithClassificationHead(nn.Module):
    def __init__(self, model, input_dim, ff_dim, dropout_p=0.1):
        super(ModelWithClassificationHead, self).__init__()
        self.model = model
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(ff_dim, 1),
        )

    def forward(
        self, 
        input_ids,  # (BATCH_SIZE, MAX_LEN)
    ):
        x = self.model(input_ids)  # (BATCH_SIZE, EMBED_DIM)
        return torch.sigmoid(self.ff(x))