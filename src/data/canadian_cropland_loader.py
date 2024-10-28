import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import copy
from functools import partial
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import default_collate
import time


class channelNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, *args, **kwargs):
        x = (x - self.mean) / self.std
        return x

class CroplandDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=None, root_dir=None):
        self.root = Path(root_dir) / "Canadian_Cropland_preprocessed.pt"
        self.channels = channels
        self.split = split
        self.transform = transform
        data = torch.load(self.root)
        if split == 'train':
            all_x = []
            all_y = []
            for year in data.keys():
                x = data[year]['train_imgs']
                y = data[year]['train_labels']
                if x.shape[1] == 13:
                    split_channels = [0,1,2,3,4,5,6,7,8,9,11,12]
                    x = copy.deepcopy(x[:,split_channels,:,:])
                else:
                    x = copy.deepcopy(x)
                all_x.append(x)
                all_y.append(y)
            self.data_x = torch.cat(all_x, dim=0)
            self.data_y = torch.cat(all_y, dim=0)
        elif split == 'test':
            all_x = []
            all_y = []
            for year in data.keys():
                x = data[year]['test_imgs']
                y = data[year]['test_labels']
                if x.shape[1] == 13:
                    split_channels = [0,1,2,3,4,5,6,7,8,9,11,12]
                    x = copy.deepcopy(x[:,split_channels,:,:])
                else:
                    x = copy.deepcopy(x)
                all_x.append(x)
                all_y.append(y)
            self.data_x = torch.cat(all_x, dim=0)
            self.data_y = torch.cat(all_y, dim=0)
        elif split == 'val':
            all_x = []
            all_y = []
            for year in data.keys():
                x = data[year]['val_imgs']
                y = data[year]['val_labels']
                if x.shape[1] == 13:
                    split_channels = [0,1,2,3,4,5,6,7,8,9,11,12]
                    x = copy.deepcopy(x[:,split_channels,:,:])
                else:
                    x = copy.deepcopy(x)
                all_x.append(x)
                all_y.append(y)
            self.data_x = torch.cat(all_x, dim=0)
            self.data_y = torch.cat(all_y, dim=0)
        del data

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        if self.transform is not None:
            x = self.transform(x)
        x = x[self.channels,:,:].float()
        return x, y

    def __len__(self):
        return len(self.data_y)
    

def load_cropland(channels=[0,1,2,3,4,5,6,7,8,9,10,11], batch_size=16, root_dir = "data/Canadian_Cropland_preprocessed.pt"):
    cutmix = v2.CutMix(num_classes=10, alpha=1.0)
    mixup = v2.MixUp(num_classes=10, alpha=0.8)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))
    
    root_dir = Path(root_dir) / "CROMA_benchmarks"
    mean_std = torch.load(root_dir / "Canadian_Cropland" / "mean_std.pt")
    mean = mean_std['mean']
    std = mean_std['std']
    train_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),
    ])
    test_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
    ])
    train_dataset = CroplandDataset('train', channels=channels, transform=train_transform, root_dir=root_dir)
    val_dataset = CroplandDataset('val', channels=channels, transform=test_transform, root_dir=root_dir)
    test_dataset = CroplandDataset('test', channels=channels, transform=test_transform, root_dir=root_dir)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    sampler_test = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=5, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_test, num_workers=5, drop_last=True, pin_memory=True)
    
    return train_loader, test_loader, val_loader