import torch
from torchvision.datasets import CIFAR10
from typing import Callable, Optional

import torchvision
import torchvision.transforms as transforms

DATA_DESC = {
    'data': 'cifar10_score',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465],
    'std': [0.2023, 0.1994, 0.2010],
}

class CIFAR10WithScores(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            score_file: str = None,
    ):
        super().__init__(root, train, transform, target_transform, download)

        self.score_file = score_file
        self.data_with_scores = torch.load(self.score_file)

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)

        # for batch idx, get the corresponding data with scores
        data_with_score = self.data_with_scores[index]
        image_with_score, score = data_with_score.chunk(2)

        return image_with_score, score, target


def load_cifar10_score(data_dir, use_augmentation='none'):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset.
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    # if use_augmentation == 'base':
    #     train_transform = transforms.Compose(
    #         [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5),
    #          transforms.ToTensor()])
    # else:
    train_transform = test_transform

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset