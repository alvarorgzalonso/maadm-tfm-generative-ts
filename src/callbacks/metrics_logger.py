import pandas as pd
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import classification_report, confusion_matrix

class MetricsLogger(Callback):
    def __init__(self, metrics_file_path="metrics_logger.csv", report_file_path="classification_report.txt"):
        self.metrics_file_path = metrics_file_path
        self.report_file_path = report_file_path
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

    def on_train_end(self, trainer, pl_module):
        print(f"Logging metrics to file: {self.metrics_file_path}")
        pd.DataFrame(self.metrics).to_csv(self.metrics_file_path, index=False)
        all_preds = []
        all_labels = []
        for batch in trainer.datamodule.val_dataloader():
            inputs, labels = batch["input"], batch["label"]
            inputs, labels = inputs.cuda(), labels.cuda()
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
