from torch.utils.data import Dataset
from typing import Callable, Optional
from os.path import join
from os import listdir
import numpy as np



class CEDataset(Dataset):
    def __init__(
            self,
            root: str,
            dataset_folder: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
        ):
        super().__init__()
        self.data_path = join(root, dataset_folder)
        # Filter and list only files with the .npz extension
        npz_files = [np.load(join(self.data_path, file)) for file in listdir(self.data_path) if file.endswith('.npz')]
        self.image = np.concatenate([f["image"] for f in npz_files], axis=0)
        self.label = np.concatenate([f["label"] for f in npz_files], axis=0)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        img = self.image[index]
        if self.transform is not None:
            img = self.transform(img)

        target = self.label[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.image.shape[0]


