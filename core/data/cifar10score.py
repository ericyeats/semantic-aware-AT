from .dataset_with_scores import DatasetWithScores

import torchvision
import torchvision.transforms as transforms


DATA_DESC = {
    'data': 'cifar10score',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465], 
    'std': [0.2023, 0.1994, 0.2010],
}


def load_cifar10score(data_dir, use_augmentation='base', time=None, n_mc_samples=None):
    """
    Returns CIFAR10scores train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    assert time is not None
    assert n_mc_samples is not None

    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5)])
    else: 
        train_transform = None # test_transform
    
    dataset_folder = "score_t{:1.2f}_mc{}".format(time, n_mc_samples)
    train_dataset = DatasetWithScores(root=data_dir, dataset_folder=dataset_folder, transform=train_transform, target_transform=None, \
                                      data_score_file="data_with_scores.pt", labels_file="labels.pt")
    test_dataset = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=False, download=True, transform=test_transform)    
    return train_dataset, test_dataset