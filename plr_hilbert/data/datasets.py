
from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_loaders(name: str = "mnist",
                data_dir: str = "./data",
                batch_size: int = 128,
                val_split: float = 0.1,
                num_workers: int = 2,
                seed: int = 42) -> Dict[str, DataLoader]:
    name = name.lower()
    g = torch.Generator().manual_seed(seed)

    if name == "mnist":
        tfm_train = transforms.Compose([transforms.ToTensor()])
        tfm_test = transforms.Compose([transforms.ToTensor()])
        full_train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm_train)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm_test)
        in_shape = (1, 28, 28)
        num_classes = 10
    elif name in ["fashion_mnist", "fmnist", "fashion-mnist"]:
        tfm_train = transforms.Compose([transforms.ToTensor()])
        tfm_test = transforms.Compose([transforms.ToTensor()])
        full_train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm_train)
        test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm_test)
        in_shape = (1, 28, 28)
        num_classes = 10
    elif name == "cifar10":
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        tfm_test = transforms.Compose([transforms.ToTensor()])
        full_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm_train)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm_test)
        in_shape = (3, 32, 32)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {name}")

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train, val = random_split(full_train, [n_train, n_val], generator=g)

    loaders = {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "meta":  {"in_shape": in_shape, "num_classes": num_classes}
    }
    return loaders
