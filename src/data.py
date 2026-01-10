import torch 
import torchvision 
import torchvision.transforms as transforms
import os, random
from PIL import Image
import pickle 
from torch.utils.data.distributed import DistributedSampler

def get_transforms(image_size, data_name = "cifar10", algo='supcon'):
    if data_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif data_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    
    s = 0.5

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
        transforms.RandomGrayscale(p = 0.2),
        transforms.ToTensor(),
    ])

    train_transforms_mlp = transforms.Compose([transforms.RandomResizedCrop(image_size),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])

    test_transforms = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize(mean = mean, std = std)
    ])

    if algo == 'simsiam' or algo == 'byol' or algo == "vicreg":
        train_transforms = transforms.Compose([
            train_transforms,
            transforms.Normalize(mean = mean, std = std)
        ])

        train_transforms_mlp = transforms.Compose([
            train_transforms_mlp,
            transforms.Normalize(mean = mean, std = std)
        ])

        test_transforms = transforms.Compose([
            test_transforms,
            transforms.Normalize(mean = mean, std = std)
        ])

    return train_transforms, train_transforms_mlp, test_transforms

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

        if self.algo == "simclr" or self.algo == "dial" or self.algo == "dare" or self.algo == "supcon" or self.algo == "simsiam" or self.algo == "byol" or self.algo == "moco" or self.algo == "barlow_twins" or self.algo == "vicreg":
            img1 = self.target_transform(image)
            img2 = self.target_transform(image)
            return img1, img2, label 
        
        return self.target_transform(image), label # if not triplet or simclr
        

def Cifar100DataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']
    algo = kwargs['algo']

    train_transforms, train_transforms_mlp, test_transforms = get_transforms(image_size, data_name = "cifar100", algo=algo)

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_dataset = DataCifar(
        algo = algo, data_name = "cifar100", 
        data_dir = data_dir, target_transform = train_transforms)

    train_dataset_mlp = torchvision.datasets.CIFAR100(
        data_dir,
        transform = train_transforms_mlp,
        train = True,
        download = True
    )
    
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

    train_dl_mlp = torch.utils.data.DataLoader(
        train_dataset_mlp,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset_mlp) if distributed else None 
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers= num_workers
    )

    return train_dl, train_dl_mlp, test_dl, train_dataset, test_dataset

def Cifar10DataLoader(**kwargs):
    image_size = kwargs['image_size']
    data_dir = kwargs['data_dir']
    algo = kwargs['algo']

    train_transforms, train_transforms_mlp, test_transforms = get_transforms(image_size, data_name = "cifar10", algo=algo)

    distributed = kwargs['distributed']
    num_workers = kwargs['num_workers']

    train_dataset = DataCifar(
        algo = algo, data_name = "cifar10", 
        data_dir = data_dir, target_transform = train_transforms)

    train_dataset_mlp = torchvision.datasets.CIFAR10(
        data_dir,
        transform = train_transforms_mlp,
        train = True,
        download = True
    )
    
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

    train_dl_mlp = torch.utils.data.DataLoader(
        train_dataset_mlp,
        batch_size = kwargs['batch_size'],
        shuffle=False if distributed else True,
        pin_memory=True,
        num_workers = num_workers,
        sampler = DistributedSampler(train_dataset_mlp) if distributed else None 
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle=True,
        pin_memory=True,
        num_workers= num_workers
    )

    return train_dl, train_dl_mlp, test_dl, train_dataset, test_dataset