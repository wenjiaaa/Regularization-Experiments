import torch
import torch.utils.data as Data
import torchvision
import os




# DOWNLOAD_IMAGENET = True

class DataLoader():
    def __init__(self, dataset, BATCH_SIZE):
        self.dataset = dataset
        self.BATCH_SIZE = BATCH_SIZE

    def MNIST(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),  # to 0-1
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1-1
        ])
        if not(os.path.exists('data/MNIST')) or not os.listdir('data/MNIST'):
               # not mnist dir or mnist is empyt dir
               DOWNLOAD_MNIST = True
        else:
               DOWNLOAD_MNIST = False
        train_data = torchvision.datasets.MNIST(
            root='data/MNIST',
            download=DOWNLOAD_MNIST,
            transform=transform,
            train=True
        )  # (60000, 28, 28)
        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # how many child process to load data
        )
        test_data = torchvision.datasets.MNIST(
            root='data/MNIST',
            train=False,
            download=DOWNLOAD_MNIST,
            transform=transform,
        )  # (10000, 28, 28)
        test_loader = Data.DataLoader(
            dataset=test_data,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=2,  # how many child process to load data
        )
        return train_loader, test_loader

    def Fashion_MNIST(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),  # to 0-1
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1-1
        ])
       
        if not(os.path.exists('data/Fashion_MNIST')) or not os.listdir('data/Fashion_MNIST'):
               # not mnist dir or mnist is empyt dir
               DOWNLOAD_Fashion_MNIST = True
        else:
               DOWNLOAD_Fashion_MNIST = False
        train_data = torchvision.datasets.FashionMNIST(
            root='data/Fashion_MNIST',
            download=DOWNLOAD_Fashion_MNIST,
            transform=transform,
            train=True
        )  # (60000, 28, 28)
        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # how many child process to load data
        )
        test_data = torchvision.datasets.FashionMNIST(
            root='data/Fashion_MNIST',
            train=False,
            download=DOWNLOAD_Fashion_MNIST,
            transform=transform,
        )  # (10000, 28, 28)
        test_loader = Data.DataLoader(
            dataset=test_data,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=2,  # how many child process to load data
        )
        return train_loader, test_loader

    def SVHN(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),  # to 0-1
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1-1
        ])
        if not(os.path.exists('data/SVHN')) or not os.listdir('data/SVHN'):
               # not mnist dir or mnist is empyt dir
               DOWNLOAD_SVHN = True
        else:
               DOWNLOAD_SVHN = False
        train_data = torchvision.datasets.SVHN(
            root='data/SVHN',
            download=DOWNLOAD_SVHN,
            transform=transform,
            split='train',
        )  # (60000, 32, 32)
        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # how many child process to load data
        )
        test_data = torchvision.datasets.SVHN(
            root='data/SVHN',
            split='test',
            download=DOWNLOAD_SVHN,
            transform=transform,
        )  # (10000, 32, 32)
        test_loader = Data.DataLoader(
            dataset=test_data,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=2,  # how many child process to load data
        )
        return train_loader, test_loader

    def CIFAR10(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),  # to 0-1
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1-1
        ])
        if not(os.path.exists('data/CIFAR10')) or not os.listdir('data/CIFAR10'):
               # not mnist dir or mnist is empyt dir
               DOWNLOAD_CIFAR10 = True
        else:
               DOWNLOAD_CIFAR10 = False
        train_data = torchvision.datasets.CIFAR10(
            root='data/CIFAR10',
            download=DOWNLOAD_CIFAR10,
            transform=transform,
            train=True
        )  # (60000, 28, 28)
        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # how many child process to load data
        )
        test_data = torchvision.datasets.CIFAR10(
            root='data/CIFAR10',
            train=False,
            download=DOWNLOAD_CIFAR10,
            transform=transform,
        )  # (10000, 32, 32)
        test_loader = Data.DataLoader(
            dataset=test_data,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=2,  # how many child process to load data
        )
        return train_loader, test_loader

    def CIFAR100(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),  # to 0-1
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1-1
        ])
        if not(os.path.exists('data/CIFAR100')) or not os.listdir('data/CIFAR100'):
               # not mnist dir or mnist is empyt dir
               DOWNLOAD_CIFAR100 = True
        else:
               DOWNLOAD_CIFAR100 = False
        train_data = torchvision.datasets.CIFAR100(
            root='data/CIFAR100',
            download=DOWNLOAD_CIFAR100,
            transform=transform,
            train=True
        )  # (50000, 32, 32)
        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # how many child process to load data
        )
        test_data = torchvision.datasets.CIFAR100(
            root='data/CIFAR100',
            train=False,
            download=DOWNLOAD_CIFAR100,
            transform=transform,
        )  # (10000, 32, 32)
        test_loader = Data.DataLoader(
            dataset=test_data,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=2,  # how many child process to load data
        )
        return train_loader, test_loader


    def load(self): # type, form of transform
        if self.dataset == 'MNIST':
            train_loader, test_loader = self.MNIST()
            return train_loader, test_loader
        elif self.dataset == 'Fashion_MNIST':
            train_loader, test_loader = self.Fashion_MNIST()
            return train_loader, test_loader
        elif self.dataset == 'SVHN':
            train_loader, test_loader = self.SVHN()
            return train_loader, test_loader
        elif self.dataset == 'CIFAR10':
            train_loader, test_loader = self.CIFAR10()
            return train_loader, test_loader
        elif self.dataset == 'CIFAR100':
            train_loader, test_loader = self.CIFAR100()
            return train_loader, test_loader
        else:
            print('Dataset name is error!')
'''
if __name__ == '__main__':
    # dataloader = DataLoader('MNIST', 50)  # 1,28,28
    dataloader = DataLoader('CIFAR100', 50)  # 1,28,28
    # dataloader = DataLoader('SVHN', 50)  # 3,32,32
    # dataloader = DataLoader('CIFAR10', 50)  # 3,32,32
    # dataloader = DataLoader('CIFAR100', 50)  # 3,32,32
    trainloader, testloader = dataloader.load()
'''