import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision

#satellite
from pathlib import Path
import tifffile
from torchvision.transforms import v2
from torch.utils.data import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class channelNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, *args, **kwargs):
        x = (x - self.mean) / self.std
        return x


class bigearthnetDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=[3,2,1], data_dir=None):
        self.root = data_dir / split
        self.channels = channels
        self.split = split
        self.transform = transform
        self.all_data = torch.load(self.root / 'all.pt')

    def __getitem__(self, index):
        x, y = self.all_data[index]
        y = y.long()
        
        if self.transform is not None:
            x = self.transform(x)

        x = x[self.channels,:,:]

        return x, y

    def __len__(self):
        return len(self.all_data)

class brickKilnDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=[0,1,2,3,4,5,6,7,8,9,10,11,12], data_dir=None):
        self.root = data_dir / split
        self.channels = channels
        self.split = split
        self.transform = transform
        self.all_data = torch.load(self.root / 'all.pt')

    def __getitem__(self, index):
        x, y = self.all_data[index]
        y = y.long()
        
        if self.transform is not None:
            x = self.transform(x)
        
        x = x[self.channels,:,:].float()

        return x, y

    def __len__(self):
        return len(self.all_data)


class EurosatDataset(torch.utils.data.Dataset):

    def __init__(self, split, transform=None, root_dir = None, channels=[0,1,2,3,4,5,6,7,8,9,11,12]):
        self.channel_mean = torch.tensor([[[[1354.4065]],[[1118.2460]],[[1042.9271]],[[947.6271]],[[1199.4736]],[[1999.7968]],[[2369.2229]],[[2296.8181]],[[732.0835]],[[12.1133]],[[1819.0067]],[[1118.9218]],[[2594.1438]]]])
        self.channel_std = torch.tensor([[[[245.7176]],[[333.0078]],[[395.0925]],[[593.7505]],[[566.4170]],[[861.1840]],[[1086.6313]],[[1117.9817]],[[404.9198]],[[4.7758]],[[1002.5877]],[[761.3032]],[[1231.5858]]]])
        self.root = root_dir
        self.channels = channels
        self.split = split
        self.transform = transform
        self.all_data = torch.load(self.root / f'{split}.pt')


    def __getitem__(self, index):
        img, target = self.all_data[index]
        img = img[self.channels]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.all_data)


class so2satDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=None, data_dir=None):
        self.root = data_dir / split
        self.channels = channels
        self.split = split
        self.transform = transform
        self.all_data = torch.load(self.root / 'all.pt')

    def __getitem__(self, index):
        x, y = self.all_data[index]
        y = y.long()
        
        if self.transform is not None:
            x = self.transform(x)
        x = x[self.channels,:,:].float() # 10*120*120
        # add 0 filled channels to 0th and 7th index
        return x, y

    def __len__(self):
        return len(self.all_data)   


class forestnetDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=None, data_dir=None):
        self.root = data_dir / split
        self.channels = channels
        self.split = split
        self.transform = transform
        self.all_data = torch.load(self.root / 'all.pt')

    def __getitem__(self, index):
        x, y = self.all_data[index]
        y = y.long()
        
        if self.transform is not None:
            x = self.transform(x)
        
        x = x[self.channels,:,:].float()

        return x, y

    def __len__(self):
        return len(self.all_data)


class pv4gerDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=None, data_dir=None):
        self.root = data_dir / split
        self.channels = channels
        self.split = split
        self.transform = transform
        self.all_data = torch.load(self.root / 'all.pt')

    def __getitem__(self, index):
        x, y = self.all_data[index]
        y = y.long()
        
        if self.transform is not None:
            x = self.transform(x)
        
        x = x[self.channels,:,:].float()

        return x, y

    def __len__(self):
        return len(self.all_data)


def load_bigearthnet(channels=[0,1,2,3,4,5,6,7,8,9,10,11], batch_size=16, root_dir="data/geobench", valid_split = 0):
    if channels == None:
        channels = [0,1,2,3,4,5,6,7,8,9,10,11]

    data_dir = Path(root_dir) / 'bigearthnet'
    mean_std = torch.load(data_dir / 'mean_std.pt')
    mean = mean_std['mean']
    std = mean_std['std']
    train_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),
        torchvision.transforms.Resize((120, 120)),\
    ])
    test_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        torchvision.transforms.Resize((120, 120)),
    ])
    train_dataset = bigearthnetDataset('train', channels=channels, transform=train_transform, data_dir=data_dir)
    test_dataset = bigearthnetDataset('test', channels=channels, transform=test_transform, data_dir=data_dir)
    print("Size of the train/val dataset:", len(train_dataset), len(test_dataset))
    if valid_split > 0:
        val_dataset = bigearthnetDataset('val', channels=channels, transform=test_transform, data_dir=data_dir)
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, sampler = sampler_val, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, sampler = sampler_test, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        return train_loader, valid_loader, test_loader

    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=5, drop_last=False, pin_memory=True)
    return train_loader, None, test_loader



    
def load_brick_kiln(channels=[0,1,2,3,4,5,6,7,8,9,10,11,12], batch_size=16, root_dir="data/geobench", valid_split = 0):
    if channels == None:
        channels = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    data_dir = Path(root_dir) / 'brickkiln'
    mean_std =  torch.load(data_dir / 'mean_std.pt')
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
    train_dataset = brickKilnDataset('train', channels=channels, transform=train_transform, data_dir=data_dir)
    test_dataset = brickKilnDataset('test', channels=channels, transform=test_transform, data_dir=data_dir)
    print(len(train_dataset), len(test_dataset))
    if valid_split > 0:
        val_dataset = brickKilnDataset('val', channels=channels, transform=test_transform, data_dir=data_dir)
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, sampler = sampler_val, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, sampler = sampler_test, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        return train_loader, valid_loader, test_loader
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=5, drop_last=False, pin_memory=True)
    return train_loader, None, test_loader


def load_EuroSAT(channels=[0,1,2,3,4,5,6,7,8,9,10,11,12], batch_size=16, root_dir="data/geobench", valid_split = 0):
    if channels == None:
        channels = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    data_dir = Path(root_dir) / 'EuroSAT_MS'
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(90),
        torchvision.transforms.Resize((64, 64)),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
    ])
    train_dataset = EurosatDataset('train', channels=channels, transform=train_transform, root_dir=data_dir)
    test_dataset = EurosatDataset('test', channels=channels, transform=test_transform, root_dir=data_dir)
    print(len(train_dataset), len(test_dataset))
    if valid_split > 0:
        val_dataset = EurosatDataset('val', channels=channels, transform=test_transform, root_dir=data_dir)
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, sampler = sampler_val, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, sampler = sampler_test, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        return train_loader, valid_loader, test_loader
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=5, drop_last=False, pin_memory=True)
    return train_loader, None, test_loader


    
def load_so2sat(channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], batch_size=16, root_dir="data/geobench", valid_split = 0):
    if channels == None:
        channels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    data_dir = Path(root_dir) / 'so2sat'
    mean_std =  torch.load(data_dir / 'mean_std.pt')
    mean = mean_std['mean']
    std = mean_std['std']
    train_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),
        torchvision.transforms.Resize((64, 64)),
    ])
    test_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        torchvision.transforms.Resize((64, 64)),
    ])
    train_dataset = so2satDataset('train', channels=channels, transform=train_transform, data_dir=data_dir)
    test_dataset = so2satDataset('test', channels=channels, transform=test_transform, data_dir=data_dir)
    print(len(train_dataset), len(test_dataset))
    if valid_split > 0:
        val_dataset = so2satDataset('val', channels=channels, transform=test_transform, data_dir=data_dir)
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, sampler = sampler_val, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, sampler = sampler_test, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        return train_loader, valid_loader, test_loader
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=5, drop_last=False, pin_memory=True)
    return train_loader, None, test_loader


    
def load_forestnet(channels=[0,1,2,3,4,5], batch_size=16, root_dir="data/geobench", valid_split = 0):
    if channels == None:
        channels = [0,1,2,3,4,5]
    data_dir = Path(root_dir) / 'forestnet'
    mean_std =  torch.load(data_dir / 'mean_std.pt')
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
    train_dataset = forestnetDataset('train', channels=channels, transform=train_transform, data_dir=data_dir)
    test_dataset = forestnetDataset('test', channels=channels, transform=test_transform, data_dir=data_dir)
    print(len(train_dataset), len(test_dataset))
    if valid_split > 0:
        val_dataset = forestnetDataset('val', channels=channels, transform=test_transform, data_dir=data_dir)
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, sampler = sampler_val, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, sampler = sampler_test, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        return train_loader, valid_loader, test_loader
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=5, drop_last=False, pin_memory=True)
    return train_loader, None, test_loader
    

def load_pv4ger(channels=[0,1,2], batch_size=16, root_dir="data/geobench", valid_split = 0):
    if channels == None:
        channels = [0,1,2]
    data_dir = Path(root_dir) / 'pv4ger'
    mean_std =  torch.load(data_dir / 'mean_std.pt')
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
    train_dataset = pv4gerDataset('train', channels=channels, transform=train_transform, data_dir=data_dir)
    test_dataset = pv4gerDataset('test', channels=channels, transform=test_transform, data_dir=data_dir)
    print(len(train_dataset), len(test_dataset))
    if valid_split > 0:
        val_dataset = pv4gerDataset('val', channels=channels, transform=test_transform, data_dir=data_dir)
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, sampler = sampler_val, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, sampler = sampler_test, batch_size=batch_size, num_workers=5, drop_last=False, pin_memory=True)
        return train_loader, valid_loader, test_loader
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=5, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=5, drop_last=False, pin_memory=True)
    return train_loader, None, test_loader


def split_dataset(train_dataset, valid_split):
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_split * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler