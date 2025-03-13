import torch 
import torchvision 
import torchvision.transforms as transforms
import os 
from src.dataset.tree import load_distances
from PIL import Image
import pickle 
from torch.utils.data.distributed import DistributedSampler

def Cifar100DataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']

    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.6),
        transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_dataset = torchvision.datasets.CIFAR100(
        data_dir, 
        transform = train_transforms,
        train=True,
        download = True)
    
    test_dataset = torchvision.datasets.CIFAR100(
        data_dir, 
        transform=test_transforms,
        train=False,
        download=True
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset) if distributed else None 
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers= num_workers
    )

    return train_dl, test_dl, train_dataset, test_dataset