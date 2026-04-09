from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_cifar10_datasets(data_dir: str):
    data_root = Path(data_dir)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_transform)
    return train_set, val_set


def build_flowers102_datasets(data_dir: str, image_size: int = 224):
    data_root = Path(data_dir)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_set = datasets.Flowers102(root=data_root, split="train", download=True, transform=train_transform)
    val_set = datasets.Flowers102(root=data_root, split="val", download=True, transform=eval_transform)
    return train_set, val_set


def build_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_set, val_set = build_cifar10_datasets(data_dir)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def build_classification_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    dataset_key = dataset_name.lower()
    if dataset_key == "cifar10":
        train_set, val_set = build_cifar10_datasets(data_dir)
    elif dataset_key == "flowers102":
        train_set, val_set = build_flowers102_datasets(data_dir, image_size=image_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
