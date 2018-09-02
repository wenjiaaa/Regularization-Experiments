#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 17:49:13 2018

@author: wenjia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:30:28 2018

@author: wenjia
"""
import matplotlib
# Force matplotlib to not use any Xwindows backend. 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import optim
import torch.nn as nn
from VGG import VGG
from dataloader import DataLoader
# from FcNet import FcNet
from train import Trainer
import argparse, time

import numpy as np
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='VGG11', help='types of networks,  VGG11,VGG13,VGG16,VGG19')
parser.add_argument('--dataset', type=str, default='MNIST', help='types of dataset, MNIST, Fashion_MNIST, SVHN, CIFAR10, CIFAR100')
parser.add_argument('--channels', type=int, default=1, help='channels of picture')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--lr', type = float, default= 1e-3, help='learning rate')
parser.add_argument('--epoches', type=int, default=20, help='epochs of training')
parser.add_argument('--batchsize', type=int, default=64, help='batch size')

#parser.add_argument('--keep_prob',type = float,default = 0.5,help = 'keep probality for dropout')
opt = parser.parse_args()
print(opt)
criterion = nn.CrossEntropyLoss(size_average=False)
dataloader = DataLoader(opt.dataset, opt.batchsize)
trainloader, testloader = dataloader.load()

all_test_loss = []
#all_train_loss = []
all_test_acc = []
#all_train_acc = []
for i in range(10):
    p = i/10
    model = VGG(vgg_name = opt.net, Fc_BN = False, Conv_BN = True , drop_prob = p,channels= opt.channels,num_classes=opt.num_classes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    trainer = Trainer(device, model, criterion,'no_reg')
    print('### dorp_prob = ' + str(p) + ' ### starting train......')
    test_loss, test_acc= trainer.train(opt.epoches, trainloader, testloader, optimizer)
    all_test_acc.append(test_acc[-1])
    all_test_loss.append(test_loss[-1])
    #all_train_acc.append(epoch_train_acc)
    #all_train_loss.append(epoch_train_loss)
    print('finish train!')
print('loss: {}  \nacc : {}'.format(all_test_loss,all_test_acc))
f = open(opt.dataset+'_best_p.txt','w')
f.write(str(all_test_loss))
f.write('\n')
f.write(str(all_test_acc))
f.close()

# plot
''''
step = np.arange(0,10)/10
plt.figure(figsize = (10,4))
plt.subplot(121)
plt.plot(step,all_test_loss,label = 'test_loss')
plt.plot(step,all_train_loss,label = 'train_loss')
plt.xlabel('drop_prob')
plt.ylabel('loss')
plt.legend()

plt.subplot(122)
plt.plot(step,all_test_acc,label = 'test_acc')
plt.plot(step,all_train_acc,label = 'train_acc')
plt.xlabel('drop_prob')
plt.ylabel('accuracy')
plt.legend()

plt.savefig(opt.dataset+'_best_p.png')
'''