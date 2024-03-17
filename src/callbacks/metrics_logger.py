import pandas as pd
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import classification_report, confusion_matrix
import torch

class MetricsLogger(Callback):
    def __init__(self, metrics_path="metrics.csv", report_path="classification_report.txt"):
        self.metrics_path = metrics_path
        self.report_path = report_path
        self.metrics = []

    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch_metrics = {
            "epoch": trainer.current_epoch,
            "train_loss": metrics.get("train_loss_epoch").item(),
            "val_loss": metrics.get("val_loss_epoch").item(),
            "train_f1_score": metrics.get("train_f1_score_epoch").item(),
            "val_f1_score": metrics.get("val_f1_score_epoch").item(),
        }
        self.metrics.append(epoch_metrics)
        pd.DataFrame(self.metrics).to_csv(self.metrics_path, index=False)

    def on_train_end(self, trainer, pl_module):
        all_preds = []
        all_labels = []
        for batch in trainer.datamodule.val_dataloader():
            inputs, labels = batch["input"], batch["label"]
            outputs = pl_module.model(inputs)
            preds = outputs.argmax(dim=1)
            labels = labels.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        report = classification_report(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        with open(self.report_path, "w") as f:
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
        
        print(report)
        print("Confusion Matrix:")
        print(conf_matrix)
