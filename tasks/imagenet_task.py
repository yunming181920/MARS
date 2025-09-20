import random

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import Subset
import numpy as np
from models.rexnetv1 import ReXNetV1
from tasks.task import Task
import os


class ImagenetTask(Task):

    def load_data(self):
        self.load_imagenet_data()
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            split = min(self.params.fl_total_participants / 20, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            train_loaders = [self.get_train(indices) for pos, indices in
                             indices_per_participant.items()]
        else:
            # sample indices for participants that are equally
            # split to 500 images per participant
            split = min(self.params.fl_total_participants / 20, 1)

            random_index=np.random.choice(len(self.train_dataset),len(self.train_dataset),replace=False)
            self.train_dataset = Subset(self.train_dataset,random_index)
            all_range = list(range(int(len(self.train_dataset) * split)))
            random.shuffle(all_range)
            train_loaders = [self.get_train_old(all_range, pos)
                             for pos in
                             range(self.params.fl_total_participants)]
        self.fl_train_loaders = train_loaders
        return

    def load_imagenet_data(self):

        train_transform = transforms.Compose([
            #
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

        self.train_dataset = torchvision.datasets.ImageFolder('../data/train',
            train_transform)

        self.test_dataset = torchvision.datasets.ImageFolder(
            '../data/val',
            test_transform)
        sample_train_size=len(self.train_dataset)//50
        sample_test_size=len(self.test_dataset)//10
        train_subset_indices=np.random.choice(len(self.train_dataset),sample_train_size,replace=False)
        test_subset_indices=np.random.choice(len(self.test_dataset),sample_test_size,replace=False)
        self.train_dataset=Subset(self.train_dataset, train_subset_indices)
        self.test_dataset=Subset(self.test_dataset, test_subset_indices)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=64,
                                       shuffle=True,  pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=64,
                                      shuffle=False, pin_memory=True)

    def build_model(self) -> None:
        return ReXNetV1()
