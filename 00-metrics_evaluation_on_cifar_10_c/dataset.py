import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from utils import (
    BATCH_SIZE,
    CIFAR10_ROOT,
    CIFAR10C_ROOT,
    CIFAR10C_CORRUPTION,
    CIFAR10C_SEVERITY,
    CLEAN_CLIENTS
)

class CIFAR10C(Dataset):
    def __init__(self, root, corruption, severity=None, transform=None):
        """
        CIFAR-10-C Dataset

        Parameters
        ----------
        root : str
            Path to CIFAR-10-C directory
        corruption : str
            Corruption type (e.g. 'gaussian_noise')
        severity : int or None
            If int in [1,5]: load that severity only
            If None: load all severities (1 to 5)
        transform : callable
            Image transformation
        """
        self.transform = transform

        images = np.load(f"{root}/{corruption}.npy", mmap_mode="r")
        labels = np.load(f"{root}/labels.npy")

        if severity is None:
            # Use all severities (50,000 samples)
            self.images = images
            self.labels = labels
        else:
            assert 1 <= severity <= 5, "Severity must be in [1, 5]"
            start = (severity - 1) * 10000
            end = severity * 10000
            self.images = images[start:end]
            self.labels = labels[start:end]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

GLOBAL_DATASET = None

def get_dataset():
    global GLOBAL_DATASET
    if GLOBAL_DATASET is None:
        transform = transforms.Compose([transforms.ToTensor()])
        GLOBAL_DATASET = torchvision.datasets.CIFAR10(
            root=CIFAR10_ROOT,
            train=True,
            download=True,
            transform=transform,
        )
    return GLOBAL_DATASET

def load_dataset(client_id):
    transform = transforms.Compose([transforms.ToTensor()])

    if client_id < CLEAN_CLIENTS:
        dataset = get_dataset()
        indices = np.array_split(np.arange(len(dataset)), CLEAN_CLIENTS)
        subset = Subset(dataset, indices[client_id])

    else:
        dataset = CIFAR10C(
            root=CIFAR10C_ROOT,
            corruption=CIFAR10C_CORRUPTION,
            severity=CIFAR10C_SEVERITY,
            transform=transform
        )
        indices = np.array_split(np.arange(len(dataset)), CLEAN_CLIENTS)
        subset = Subset(dataset, indices[client_id - CLEAN_CLIENTS])

    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
