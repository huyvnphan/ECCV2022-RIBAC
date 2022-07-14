import torch
from torch.optim.lr_scheduler import MultiStepLR


def shared_configure_optimizers(self):
    if self.hparams.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    elif self.hparams.optimizer == "Adam":
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
    if self.hparams.optimizer == "SGD":
        milestones = [
            int(0.25 * self.hparams.max_epochs),
            int(0.50 * self.hparams.max_epochs),
            int(0.75 * self.hparams.max_epochs),
        ]
    elif self.hparams.optimizer == "Adam":
        milestones = [
            int(0.50 * self.hparams.max_epochs),
            int(0.75 * self.hparams.max_epochs),
        ]
    scheduler = {
        "scheduler": MultiStepLR(optimizer, milestones=milestones, gamma=0.1),
        "interval": "epoch",
    }
    return [optimizer], [scheduler]
