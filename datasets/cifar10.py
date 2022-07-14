import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.raw_mean = [0.4914, 0.4822, 0.4465]
        self.raw_std = [0.2470, 0.2435, 0.2616]
        self.mean = torch.tensor(self.raw_mean).view(3, 1, 1)
        self.std = torch.tensor(self.raw_std).view(3, 1, 1)
        self.lower_limit = (0.0 - self.mean) / self.std
        self.upper_limit = (1.0 - self.mean) / self.std

        self.image_size = (3, 32, 32)
        self.no_classes = 10

        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def clamp_valid_range(self, image):
        assert len(image.size()) == 4
        self.upper_limit = self.upper_limit.to(image.device)
        self.lower_limit = self.lower_limit.to(image.device)
        return torch.max(
            torch.min(image, self.upper_limit.unsqueeze(0)),
            self.lower_limit.unsqueeze(0),
        )

    def scale(self, image):
        assert len(image.size()) == 4
        self.std = self.std.to(image.device)
        return image / self.std.unsqueeze(0)

    def normalize(self, image):
        assert len(image.size()) == 4
        self.std = self.std.to(image.device)
        self.mean = self.mean.to(image.device)
        return (image - self.mean.unsqueeze(0)) / self.std.unsqueeze(0)

    def unnormalize(self, image):
        assert len(image.size()) == 4
        self.std = self.std.to(image.device)
        self.mean = self.mean.to(image.device)
        return (image * self.std.unsqueeze(0)) + self.mean.unsqueeze(0)

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(self.image_size[1], padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.raw_mean, self.raw_std),
            ]
        )
        dataset = CIFAR10(root=self.data_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.CenterCrop(self.image_size[1]),
                T.ToTensor(),
                T.Normalize(self.raw_mean, self.raw_std),
            ]
        )
        dataset = CIFAR10(root=self.data_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
