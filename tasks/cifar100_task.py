import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
# from torchvision.models import resnet18
from models.resnet import resnet18
from tasks.task import Task
from torchvision.datasets import CIFAR100
import random

class CIFAR100Task(Task):
    normalize = transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762))

    def load_data(self):
        self.load_cifar_data()
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            split = min(self.params.fl_total_participants / 100, 1)
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
            split = min(self.params.fl_total_participants / 100, 1)
            # split = 1
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)
            train_loaders = [self.get_train_old(all_range, pos)
                             for pos in
                             range(self.params.fl_total_participants)]
        self.fl_train_loaders = train_loaders
        return

    def load_cifar_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])
        self.train_dataset = CIFAR100(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True,
                                       num_workers=0)
        self.test_dataset = CIFAR100(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish',
                        'flatfish', 'ray', 'shark', 'trout', 'orchid', 'poppy', 'rose',
                        'sunflower', 'tulip', 'bottle', 'bowl', 'can', 'cup', 'plate',
                        'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper', 'clock',
                        'keyboard', 'lamp', 'telephone', 'television', 'bed', 'chair',
                        'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly',
                        'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger',
                        'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper',
                        'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel',
                        'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox',
                        'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster',
                        'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man',
                        'woman', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                        'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple_tree',
                        'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
                        'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
                        'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor')
        return True

    def build_model(self) -> nn.Module:
        # model = resnet18(pretrained=True,
        #                  num_classes=len(self.classes))
        # model.conv1 = nn.Conv2d(model.conv1.in_channels,
        #                         model.conv1.out_channels,
        #                         3, 1, 1, bias=False)
        # model.maxpool = nn.Identity()
        # model.fc = nn.Linear(model.fc.in_features, 100)
        model = resnet18(pretrained=True)

        # Modify the model for CIFAR-100
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 100)  # Change the output size to 100 for CIFAR-100
        # for name,param in model.state_dict().items():
        #     print(name)
        #     print(param.shape)
        # exit(0)
        return model
    #
    # def build_model(self) -> nn.Module:
    #     # model = resnet18(pretrained=True,
    #     #                  num_classes=len(self.classes))
    #     # model.conv1 = nn.Conv2d(model.conv1.in_channels,
    #     #                         model.conv1.out_channels,
    #     #                         3, 1, 1, bias=False)
    #     # model.maxpool = nn.Identity()
    #     # model.fc = nn.Linear(model.fc.in_features, 100)
    #     model = resnet18(pretrained=True)
    #
    #     # Modify the model for CIFAR-100
    #     # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #     # model.maxpool = nn.Identity()
    #     model.fc = nn.Linear(model.fc.in_features, 100)  # Change the output size to 100 for CIFAR-100
    #     # for name,param in model.state_dict().items():
    #     #     print(name)
    #     #     print(param.shape)
    #     # exit(0)
    #     return model