import torch
from torch import nn

class ModelWithClassificationHead(nn.Module):
    def __init__(self, model, input_dim, ff_dim, num_classes, dropout_p=0.1):
        super(ModelWithClassificationHead, self).__init__()
        self.model = model
        # Cambiamos la última capa para tener num_classes unidades de salida
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(ff_dim, num_classes),  # Ajustado para soportar multiclase
        )
        self.num_classes = num_classes

    def forward(self, input_ids):
        x = self.model(input_ids)  # (BATCH_SIZE, EMBED_DIM)
        logits = self.ff(x)  # Aquí obtenemos los logits
        
        # Si es una clasificación binaria (num_classes = 1), aplicamos sigmoid.
        # Para multiclase, devolvemos los logits directamente para usar con CrossEntropyLoss fuera del modelo.
        if self.num_classes == 2:
            return torch.sigmoid(logits)
        else:
            return logits  # Para multiclase, devolvemos los logits directamente.
