import pandas as pd
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import classification_report, confusion_matrix
import torch

class MetricsLogger(Callback):
    def __init__(self, metrics_file_path="metrics_logger.csv", report_file_path="classification_report.txt"):
        self.metrics_file_path = metrics_file_path
        self.report_file_path = report_file_path
        self.metrics = []

    def on_train_end(self, trainer, pl_module):
        all_preds = []
        all_labels = []
        for batch in trainer.datamodule.val_dataloader():
            inputs, labels = batch["input"], batch["label"]
            #check if cuda is available
            if torch.cuda.is_available(): inputs, labels = inputs.cuda(), labels.cuda()
            labels = batch["label"].float().view(-1, trainer.datamodule.num_classes)  # (BATCH_SIZE, num_classes)
            outputs = pl_module.model(inputs)
            preds = outputs.argmax(dim=1)
            labels = labels.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        report = classification_report(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        with open(self.report_file_path, "w") as f:
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
        
        print(report)
        print("Confusion Matrix:")
        print(conf_matrix)
