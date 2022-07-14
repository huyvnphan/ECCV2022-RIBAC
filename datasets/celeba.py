import os

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class CelebA(Dataset):
    def __init__(self, root, split, transform):
        self.dataset = torchvision.datasets.CelebA(
            root=root, split=split, target_type="attr", download=False
        )
        self.list_attributes = [18, 31, 21]
        self.transform = transform
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (
            (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transform(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


class CelebADataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.raw_mean = [0.5, 0.5, 0.5]
        self.raw_std = [0.5, 0.5, 0.5]
        self.mean = torch.tensor(self.raw_mean).view(3, 1, 1)
        self.std = torch.tensor(self.raw_std).view(3, 1, 1)
        self.lower_limit = (0.0 - self.mean) / self.std
        self.upper_limit = (1.0 - self.mean) / self.std

        self.image_size = (3, 64, 64)
        self.no_classes = 8

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
                T.Resize(self.image_size[1]),
                T.RandomCrop(self.image_size[1], padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.raw_mean, self.raw_std),
            ]
        )
        dataset = CelebA(
            root=self.data_dir,
            split="train",
            transform=transform,
        )
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
                T.Resize(self.image_size[1]),
                T.CenterCrop(self.image_size[1]),
                T.ToTensor(),
                T.Normalize(self.raw_mean, self.raw_std),
            ]
        )
        dataset = CelebA(root=self.data_dir, split="valid", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
