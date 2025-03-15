import torch 
import torchvision 
import torchvision.transforms as transforms
import os, random
from PIL import Image
import pickle 
from torch.utils.data.distributed import DistributedSampler

def get_transforms(image_size):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(image_size),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayScale(p = 0.2),
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    return train_transforms, test_transforms

class DataCifar():
    def __init__(self, algo = "supcon", data_name = "cifar10", data_dir = "datasets/cifar10", target_transform = transforms.ToTensor()):
        if data_name == "cifar10":
            self.data = torchvision.datasets.CIFAR10(data_dir, train = True, download = True)
        elif data_name == "cifar100":
            self.data = torchvision.datasets.CIFAR100(data_dir, train = True, download = True)

        self.algo = algo
        self.target_transform = target_transform
        
        if self.algo == "triplet":
            len_data = len(self.data)
            data_classes = len(self.data.classes)
            self.all_data = {i:[] for i in range(data_classes)}

            for idx in range(len_data):
                self.all_data[self.data[idx][1]].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.algo == "triplet":
            anc, anc_label = self.data[idx]

            pos_idx = random.choice(self.all_data[anc_label])

            all_classes_idx = list(self.all_data.keys())
            all_classes_idx.remove(anc_label)
            neg_label = random.choice(all_classes_idx)

            neg_idx = random.choice(self.all_data[neg_label])

            pos, pos_label = self.data[pos_idx]
            neg, neg_label = self.data[neg_idx]

            anc = self.target_transform(anc)
            pos = self.target_transform(pos)
            neg = self.target_transform(neg)

            return anc, anc_label, pos, pos_label, neg, neg_label
            
        image, label = self.data[idx]

        if self.algo == "simclr":
            img1 = self.target_transform(image)
            img2 = self.target_transform(image)
            return img1, img2, label 
        
        return self.target_transform(image), label # if not triplet or simclr
        

def Cifar100DataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']
    algo = kwargs['algo']

    train_transforms, test_transforms = get_transforms(image_size)

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_dataset = DataCifar(
        algo = algo, data_name = "cifar100", 
        data_dir = data_dir, target_transform = train_transforms)
    
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

def Cifar10DataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']
    algo = kwargs['algo']

    train_transforms, test_transforms = get_transforms(image_size)

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_dataset = DataCifar(
        algo = algo, data_name = "cifar10", 
        data_dir = data_dir, target_transform = train_transforms)
    
    test_dataset = torchvision.datasets.CIFAR10(
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