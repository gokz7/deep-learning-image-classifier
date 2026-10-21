"""Data pipeline: download CIFAR-10, split train/val, apply transforms, build DataLoaders."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import Config, CIFAR10_MEAN, CIFAR10_STD
from src.utils import seed_worker, get_generator


def get_transforms():
    """Training gets augmentation; validation/test get only normalization."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # pad 4px, then crop a random 32x32 area
        transforms.RandomHorizontalFlip(),      # flip left-right with 50% chance
        transforms.ToTensor(),                  # pixels 0-255 -> 0-1, shape (3, 32, 32)
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(cfg: Config):
    """Returns train, validation and test DataLoaders."""
    train_tf, eval_tf = get_transforms()

    # Two views of the same 50k training images:
    # one WITH augmentation (for training), one WITHOUT (for validation)
    train_full_aug = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=train_tf)
    train_full_eval = datasets.CIFAR10(cfg.data_dir, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=eval_tf)

    # Fixed, seeded split: 45,000 train / 5,000 validation (no overlap)
    g = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(len(train_full_aug), generator=g).tolist()
    val_idx = indices[:cfg.val_size]
    train_idx = indices[cfg.val_size:]

    train_set = Subset(train_full_aug, train_idx)
    val_set = Subset(train_full_eval, val_idx)

    loader_args = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),       # faster CPU -> GPU copy
        persistent_workers=(cfg.num_workers > 0),  # keep workers alive between epochs
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(train_set, shuffle=True, generator=get_generator(cfg.seed), **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Undo normalization so an image can be displayed with normal colours."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)
