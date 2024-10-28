import os
import pandas as pd
import numpy as np
import warnings
import random
from glob import glob

from typing import Any, Optional, List

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio import logging
import tifffile
import time
from torchvision.transforms import v2
from torch.utils.data import default_collate
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]

class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape(13,1,1)
        self.std = torch.tensor(std).reshape(13,1,1)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = torch.clip(img, 0, 255)/255
        return img


class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
        return transforms.Compose(t)


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
           948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: Path,
                 transform: Any,
                 masked_bands: Optional[List[int]] = None,
                 channels: Optional[List[int]] = None):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        csv_path = str(csv_path)
        super().__init__(in_c=13)
        df_full = pd.read_csv(csv_path) \
            .sort_values(['category', 'location_id'])

        # Filter by category
        self.categories = CATEGORIES

        self.transform = transform
        self.channels = channels
        self.root_dir = '/data/user_data/zongzhex'

        if "train" in csv_path:
            df_full['image_path'] = f"{self.root_dir}/fmow-sentinel/fmow-sentinel/train/"+df_full['category'].astype(str)+"/"+df_full['category'].astype(str)+"_"+df_full['location_id'].astype(str)+"/"+df_full['category'].astype(str)+"_"+df_full['location_id'].astype(str)+"_"+df_full['image_id'].astype(str)+".tif"
            sub_index = random_subset_index(len(df_full), 0.1)
            self.df = df_full.iloc[sub_index].copy()
        elif "val" in csv_path:
            df_full['image_path'] = f"{self.root_dir}/fmow-sentinel/fmow-sentinel/val/"+df_full['category'].astype(str)+"/"+df_full['category'].astype(str)+"_"+df_full['location_id'].astype(str)+"/"+df_full['category'].astype(str)+"_"+df_full['location_id'].astype(str)+"_"+df_full['image_id'].astype(str)+".tif"
            self.df = df_full.copy()
        elif "test" in csv_path:
            df_full['image_path'] = f"{self.root_dir}/fmow-sentinel/fmow-sentinel/test/"+df_full['category'].astype(str)+"/"+df_full['category'].astype(str)+"_"+df_full['location_id'].astype(str)+"/"+df_full['category'].astype(str)+"_"+df_full['location_id'].astype(str)+"_"+df_full['image_id'].astype(str)+".tif"
            self.df = df_full.copy()

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        img = tifffile.imread(img_path)
        return torch.tensor(img).float().permute(2, 0, 1)  # (c, h, w)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.df.iloc[idx]

        images = self.open_image(selection['image_path'])
        labels = self.categories.index(selection['category'])
        img_as_tensor = self.transform(images)  # (c, h, w)

        if self.channels is not None:
            img_as_tensor = img_as_tensor[self.channels]
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.Resize((input_size, input_size), interpolation=interpol_mode))
            t.append(transforms.RandomHorizontalFlip())
            t.append(transforms.RandomVerticalFlip())
            t.append(transforms.RandomRotation(90))
            return transforms.Compose(t)

        t.append(SentinelNormalize(mean, std))
        t.append(
            transforms.Resize((input_size, input_size), interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        return transforms.Compose(t)


def build_fmow_dataset(root_dir, is_train: str, channels: Optional[List[int]] = None) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    root_dir = Path(root_dir) / "fmow-sentinel" / "fmow-sentinel"
    train_path = root_dir / "train.csv"
    test_path = root_dir / "test.csv"
    val_path = root_dir / "val.csv"

    if is_train == 'train':
        csv_path = train_path
    elif is_train == 'test':
        csv_path = test_path
    elif is_train == 'val':
        csv_path = val_path

    if is_train == 'train':
        is_train = True
    else:
        is_train = False

    mean = SentinelIndividualImageDataset.mean
    std = SentinelIndividualImageDataset.std
    transform = SentinelIndividualImageDataset.build_transform(is_train, 96, mean, std)
    dataset = SentinelIndividualImageDataset(csv_path, transform, masked_bands=None,
                                                channels = channels)

    return dataset

def load_fmow(batch_size = 16, channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], root_dir = "data/fmow-sentinel/fmow-sentinel/"):
    cutmix = v2.CutMix(num_classes=62, alpha=1.0)
    mixup = v2.MixUp(num_classes=62, alpha=0.8)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))
    dataset_train = build_fmow_dataset(is_train='train', channels = channels, root_dir = root_dir)
    dataset_val = build_fmow_dataset(is_train='val', channels = channels, root_dir = root_dir)
    dataset_test = build_fmow_dataset(is_train='test', channels = channels, root_dir = root_dir)

    print(f"Train dataset length: {len(dataset_train)}", 
            f"\nValidation dataset length: {len(dataset_val)}", 
            f"\nTest dataset length: {len(dataset_test)}")
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=5,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size,
        num_workers=5,
        drop_last=True,
        pin_memory=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=batch_size,
        num_workers=5,
        drop_last=True,
        pin_memory=True,
    )
    return data_loader_train, data_loader_val, data_loader_test

def random_subset_index(leng, frac, seed=34):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(leng), int(frac * leng))
    return indices