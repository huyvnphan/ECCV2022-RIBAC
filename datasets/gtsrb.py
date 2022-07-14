import os

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class GTSRBDataset(Dataset):
    base_folder = "GTSRB"

    def __init__(self, root, train=True, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root

        self.sub_directory = "trainingset" if train else "testset"
        self.csv_file_name = "training.csv" if train else "test.csv"

        csv_file_path = os.path.join(
            root, self.base_folder, self.sub_directory, self.csv_file_name
        )

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.root,
            self.base_folder,
            self.sub_directory,
            self.csv_data.iloc[idx, 0],
        )
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


class GTSRBDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        # self.raw_mean = [0.3337, 0.3064, 0.3171]
        # self.raw_std = [0.2672, 0.2564, 0.2629]
        self.raw_mean = [0.5, 0.5, 0.5]
        self.raw_std = [0.5, 0.5, 0.5]
        self.mean = torch.tensor(self.raw_mean).view(3, 1, 1)
        self.std = torch.tensor(self.raw_std).view(3, 1, 1)
        self.lower_limit = (0.0 - self.mean) / self.std
        self.upper_limit = (1.0 - self.mean) / self.std

        self.image_size = (3, 32, 32)
        self.no_classes = 43

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
                # T.RandomRotation(10),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.raw_mean, self.raw_std),
            ]
        )
        dataset = GTSRBDataset(root=self.data_dir, train=True, transform=transform)
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
        dataset = GTSRBDataset(root=self.data_dir, train=False, transform=transform)
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
