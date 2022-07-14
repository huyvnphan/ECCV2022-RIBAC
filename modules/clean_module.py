import pytorch_lightning as pl
import torch.nn as nn

from utils.accuracy import Accuracy, AverageAccuracy
from modules.share_steps import shared_configure_optimizers


class CleanModule(pl.LightningModule):
    """
    Compress model using prunning with clean data
    """

    def __init__(self, model, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.model = model

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc_clean = Accuracy()

        self.val_acc_clean = Accuracy()
        self.val_acc_average = AverageAccuracy()

    def training_step(self, batch, batch_idx):
        # Forward Pass
        image, label = batch
        pred_clean = self.model(image)

        # Loss
        loss_clean = self.criterion(pred_clean, label)

        # Accuracy
        self.train_acc_clean(pred_clean, label)

        # Log
        self.log("loss/clean", loss_clean)
        self.log("accuracy_train/clean", self.train_acc_clean)
        return loss_clean

    def validation_step(self, batch, batch_idx):
        # Forward Pass
        image, label = batch
        pred_clean = self.model(image)

        # Accuracy
        self.val_acc_clean(pred_clean, label)
        self.val_acc_average(pred_clean, pred_clean, label, label)

        # Log
        self.log("accuracy_val/clean", self.val_acc_clean)
        self.log("accuracy_val/average", self.val_acc_average)

    def test_step(self, batch, batch_idx):
        # Forward Pass
        image, label = batch
        pred_clean = self.model(image)

        # Accuracy
        self.val_acc_clean(pred_clean, label)

        # Log
        self.log("accuracy_test/clean", self.val_acc_clean)

    def configure_optimizers(self):
        return shared_configure_optimizers(self)
