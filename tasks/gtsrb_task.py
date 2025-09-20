import random
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from models.resnet import resnet18
from tasks.task import Task
import torch
import torch.nn.functional as F
class GTSRBNet(nn.Module):
    def __init__(self, num_classes):
        super(GTSRBNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(4 * 4 * 128, 512)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x
class GTSRBTask(Task):
    normalize = transforms.Normalize((0.3405, 0.3128, 0.3211),
                                     (0.2729, 0.2604, 0.2746))

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
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)
            train_loaders = [self.get_train_old(all_range, pos)
                             for pos in
                             range(self.params.fl_total_participants)]
        self.fl_train_loaders = train_loaders
        #这里产生了用于联邦学习的各个用户的train_loaders
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
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            self.normalize,
        ])
        self.train_dataset = torchvision.datasets.GTSRB(
            root=self.params.data_path,
            split='train',
            download=True,
            transform=transform_train)
        self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True)
        self.test_dataset = torchvision.datasets.GTSRB(
            root=self.params.data_path,
            split='test',
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False)

        self.classes = (
            '20_speed_limit', '30_speed_limit', '50_speed_limit', '60_speed_limit', '70_speed_limit',
            '80_speed_limit', '80_lifted', '100_speed_limit', '120_speed_limit', 'no_overtaking',
            'no_overtaking_trucks', 'right_of_way', 'right_of_way_crossing', 'give_way', 'stop',
            'no_way', 'no_way_trucks', 'no_entry', 'danger', 'bend_left', 'bend_right', 'bend',
            'uneven_road', 'slippery_road', 'road_narrows', 'construction', 'traffic_signal',
            'pedestrian_crossing', 'school_crossing', 'cycles_crossing', 'snow', 'animals', 'restriction_ends',
            'go_right', 'go_left', 'go_straight', 'go_right_or_straight', 'go_left_or_straight', 'keep_right',
            'keep_left', 'roundabout', 'restriction_ends_overtaking', 'restriction_ends_overtaking_trucks'
        )

        return True
    #这个函数完成了train_dataset的加载
    def build_model(self) -> nn.Module:
        # model = resnet18(pretrained=True,
        #                  num_classes=len(self.classes))
        # model.conv1 = nn.Conv2d(model.conv1.in_channels,
        #                         model.conv1.out_channels,
        #                         3, 1, 1, bias=False)
        # model.maxpool = nn.Identity()
        # model.fc = nn.Linear(model.fc.in_features, 100)
        #model = resnet18_gn(pretrained=False, num_classes=10)#//这里是原来的代码
        num_classes = 43  # GTSRB 数据集有 43 个类别
        model = GTSRBNet(num_classes)
        #model=ResNetS(nclasses=10)
        # # Modify the model for CIFAR-100
        # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # model.maxpool = nn.Identity()
        # model.fc = nn.Linear(model.fc.in_features, 100)  # Change the output size to 100 for CIFAR-100
        return model