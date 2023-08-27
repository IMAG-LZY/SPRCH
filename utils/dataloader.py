import torch
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR100
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from data.loader_cifar100 import CIFAR100_sample
from PIL import Image


class MyDataset(nn.Module):
    def __init__(self, data_path, data_name, txt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fp = open(txt, 'r')
        images = []
        labels = []
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append([float(l) for l in information[1:len(information)]])
        self.images = list(map(lambda x: data_path+'/'+data_name+'/'+x, images))
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        imageName = self.images[item]
        label = self.labels[item]
        image = Image.open(imageName).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = torch.FloatTensor(label)
        return image, label

    def __len__(self):
        return len(self.images)


def init_dataloader(data_path, data_name, train_list, database_list, test_list, batchSize):
    """load dataset"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_1 = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_2 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    def one_hot(nclass):
        def f(index):
            index = torch.tensor(int(index)).long()
            return torch.nn.functional.one_hot(index, nclass).float()
        return f
    if data_name == 'cifar100':
        train_dataset = CIFAR100_sample(data_path, sample_id_file='data/cifar100_sample_id_training.npy', train=True,
                                        transform=transform_1, target_transform=one_hot(100))
        database_dataset = CIFAR100(data_path, train=True, transform=transform_1, target_transform=one_hot(100))
        test_dataset = CIFAR100(data_path, train=False, transform=transform_2, target_transform=one_hot(100))
    else:
        train_dataset = MyDataset(data_path=data_path, data_name=data_name, txt=train_list, transform=transform_1)
        database_dataset = MyDataset(data_path=data_path, data_name=data_name, txt=database_list, transform=transform_2)
        test_dataset = MyDataset(data_path=data_path, data_name=data_name, txt=test_list, transform=transform_2)
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=True)
    database_loader = DataLoader(database_dataset, batch_size=batchSize, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, num_workers=8, pin_memory=True)
    print(f'train set: {len(train_dataset)}', f'database set: {len(database_dataset)}',
          f'test set: {len(test_dataset)}')
    return train_loader, database_loader, test_loader





