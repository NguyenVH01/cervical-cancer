import os
import json
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

class Mendeley():
    def __init__(self, config, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        self.data_path = config.DATA.DATA_PATH
        self.batch_size = config.DATA.BATCH_SIZE
        self.num_classes = config.MODEL.NUM_CLASSES
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Định nghĩa các phép biến đổi cho dữ liệu train với augmentation
        self.train_transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )

        # Định nghĩa các phép biến đổi cho dữ liệu val và test (không có augmentation)
        self.test_transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # Tải dữ liệu và áp dụng các phép biến đổi
        self.dataset = datasets.ImageFolder(
            root=self.data_path, transform=self.train_transform)

        # Chia dữ liệu thành train, val, test
        self.train_dataset, self.val_dataset, self.test_dataset = self.split_dataset()

        # Áp dụng transform riêng cho val và test
        self.val_dataset.dataset.transform = self.test_transform
        self.test_dataset.dataset.transform = self.test_transform

        # Tạo WeightedRandomSampler cho tập train
        self.train_sampler = self.create_sampler(self.train_dataset)

        # Tạo DataLoader cho train, val, test
        self.train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        self.val_loader = DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def split_dataset(self):
        train_size = int(self.train_ratio * len(self.dataset))
        val_size = int(self.val_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        return random_split(self.dataset, [train_size, val_size, test_size])

    def create_sampler(self, dataset):
        # Tính toán số lượng mẫu cho mỗi lớp trong tập train
        class_counts = [0] * self.num_classes
        for _, label in dataset:
            class_counts[label] += 1

        # Tính trọng số cho mỗi lớp trong tập train
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = [0] * len(dataset)
        for idx, (_, label) in enumerate(dataset):
            sample_weights[idx] = class_weights[label]

        # Khởi tạo WeightedRandomSampler cho tập train
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return sampler

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader, self.train_dataset.dataset, self.val_dataset.dataset, self.test_dataset.dataset
