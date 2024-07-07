import torch
from torch.utils.data import Dataset
from typing import Callable, Optional
from os.path import join



class DatasetWithScores(Dataset):
    def __init__(
            self,
            root: str,
            dataset_folder: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            data_score_file: str = "data_with_scores.pt",
            labels_file: str = "labels.pt"
    ):
        super().__init__()
        self.data_score_file = data_score_file
        self.labels_file = labels_file
        self.data_path = join(root, dataset_folder)
        self.data_with_scores = torch.load(join(self.data_path, self.data_score_file))
        self.labels = torch.load(join(self.data_path, self.labels_file))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        img = self.data_with_scores[index]
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data_with_scores.shape[0]


