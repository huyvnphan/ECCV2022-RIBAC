import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils.accuracy import Accuracy, AverageAccuracy
from modules.share_steps import shared_configure_optimizers


class BackdoorModule(pl.LightningModule):
    """
    Compress model using prunning with backdoor data
    """

    def __init__(self, model, backdoor, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.model = model
        self.backdoor = backdoor

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc_clean = Accuracy()
        self.train_acc_backdoor = Accuracy()
        self.train_acc_average = AverageAccuracy()

        self.val_acc_clean = Accuracy()
        self.val_acc_backdoor = Accuracy()
        self.val_acc_average = AverageAccuracy()

    def forward(self, image, label):
        backdoor_image, target = self.backdoor(image, label)

        all_image = torch.cat((image, backdoor_image), dim=0)
        all_pred = self.model(all_image)

        pred_clean = all_pred[: image.size(0)]
        pred_backdoor = all_pred[image.size(0) :]
        return pred_clean, pred_backdoor, target

    def training_step(self, batch, batch_idx):
        # Forward Pass
        image, label = batch
        pred_clean, pred_backdoor, target = self.forward(image, label)

        # Loss
        loss_clean = self.criterion(pred_clean, label)
        loss_backdoor = self.criterion(pred_backdoor, target)
        loss_total = loss_clean + loss_backdoor

        # Accuracy
        self.train_acc_clean(pred_clean, label)
        self.train_acc_backdoor(pred_backdoor, target)
        self.train_acc_average(pred_clean, pred_backdoor, label, target)

        # Log
        self.log("loss/clean", loss_clean)
        self.log("loss/backdoor", loss_backdoor)
        self.log("loss/total", loss_total, prog_bar=True)
        self.log("accuracy_train/clean", self.train_acc_clean, on_epoch=True)
        self.log("accuracy_train/backdoor", self.train_acc_backdoor, on_epoch=True)
        self.log("accuracy_train/average", self.train_acc_average, on_epoch=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        # Forward Pass
        image, label = batch
        pred_clean, pred_backdoor, target = self.forward(image, label)

        # Accuracy
        self.val_acc_clean(pred_clean, label)
        self.val_acc_backdoor(pred_backdoor, target)
        self.val_acc_average(pred_clean, pred_backdoor, label, target)

        # Log
        self.log("accuracy_val/clean", self.val_acc_clean)
        self.log("accuracy_val/backdoor", self.val_acc_backdoor)
        self.log("accuracy_val/average", self.val_acc_average)

    def test_step(self, batch, batch_idx):
        # Forward Pass
        image, label = batch
        pred_clean, pred_backdoor, target = self.forward(image, label)

        # Accuracy
        self.val_acc_clean(pred_clean, label)
        self.val_acc_backdoor(pred_backdoor, target)
        self.val_acc_average(pred_clean, pred_backdoor, label, target)

        # Log
        self.log("accuracy_test/clean", self.val_acc_clean)
        self.log("accuracy_test/backdoor", self.val_acc_backdoor)
        self.log("accuracy_test/average", self.val_acc_average)

    def configure_optimizers(self):
        return shared_configure_optimizers(self)
