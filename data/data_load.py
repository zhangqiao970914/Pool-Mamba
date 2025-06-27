import os
from PIL import Image as Image
from .data_augment import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )

    dataloader = DataLoader(
        LLIEDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    return dataloader


def valid_dataloader(path, num_workers=0, is_test=False):
    dataloader = DataLoader(
        LLIEDataset(path, is_test=is_test),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader

import os


class LLIEDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'low/'))
        self.image_list = self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'low', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'high', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        valid_extensions = ['png', 'jpg', 'jpeg', 'JPG']
        valid_files = []
        for x in lst:
            if x.split('.')[-1] in valid_extensions:
                valid_files.append(x)
        return valid_files




