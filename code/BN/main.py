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
from dataloader import DataLoader
from torch import optim
import torch.nn as nn
from VGG import VGG
from LeNet import LeNet
from train import Trainer
import argparse, time

import numpy as np
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='LeNet', help='types of networks, LeNet or VGG11,VGG13,VGG16,VGG19')
parser.add_argument('--dataset', type=str, default='MNIST', help='types of dataset, MNIST, Fashion_MNIST, SVHN, CIFAR10, CIFAR100')
parser.add_argument('--channels', type=int, default=1, help='channels of picture')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--lr', type = float, default= 1e-3, help='learning rate')
parser.add_argument('--epoches', type=int, default=10, help='epochs of training')
parser.add_argument('--batchsize', type=int, default=64, help='batch size')
opt = parser.parse_args()
print(opt)



learingrates = [opt.lr,5*opt.lr,30*opt.lr]       
criterion = nn.CrossEntropyLoss(size_average=False)

def base_model(base_model_dict,trainloader,testloader):
       modelname = list(base_model_dict.keys())[0]
       model  =base_model_dict[modelname]
       model.to(device)
       #optimizer = optim.SGD(model.parameters(), lr=opt.lr)
       optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
       trainer=Trainer(device,model,criterion)
       print('### '+modelname+' ### starting train......')
       start = time.clock()
       step_loss,step_acc = trainer.train(opt.epoches,trainloader,testloader,optimizer)
       #all_loss.append(step_loss)
       #all_acc.append(step_acc)
       end = time.clock()
       print('finish train!')
       test_loss,test_acc = trainer.test(testloader)
       print("test:  |loss {:f}   |acc {:f}   |train time {:f}".format(test_loss, test_acc,(end - start)))
       loss = {modelname:step_loss}
       acc = {modelname:step_acc}
       #results = {'step_loss':step_acc,'step_acc':step_acc,'test_loss':test_loss,'test_acc':test_acc}
       return loss,acc

def bn_model(bn_model_dict,trainloader,testloader):
       modelname = list(bn_model_dict.keys())[0]
       model = bn_model_dict[modelname]
       model.to(device)
       trainer = Trainer(device,model,criterion)
       loss = {}
       acc = {}
       for lr in learingrates:
              name = modelname+'_'+str(lr)
              optimizer = optim.Adam(model.parameters(),lr=opt.lr,betas=(0.5,0.99))
              #optimizer = optim.SGD(model.parameters(), lr=lr)
              print('### '+name+' ### starting train......')
              start = time.clock()
              step_loss,step_acc = trainer.train(opt.epoches,trainloader,testloader,optimizer)
              end = time.clock()
              print('finish train!')
              test_loss,test_acc = trainer.test(testloader)
              print("test:  |loss {:f}   |acc {:f}   |train time {:f}".format(test_loss, test_acc,(end - start)))
              
              loss[name  ] = step_loss
              #loss[name + '_test'] = test_loss
              acc[name ] = step_acc
              #acc[name + '_test'] = test_acc
       return loss,acc
       
if __name__ == '__main__':
     if opt.net == 'LeNet':
            
            base_model_dict = {opt.net:LeNet(False,opt.channels,opt.num_classes)}
            bn_model_dict = {'BN_LeNet':LeNet(True,opt.channels,opt.num_classes)}
     else:
            base_model_dict = {opt.net:VGG(opt.net,False,opt.channels,opt.num_classes)}
            bn_model_dict = {'BN_'+opt.net:VGG(opt.net,True,opt.channels,opt.num_classes)}
     dataloader = DataLoader(opt.dataset, opt.batchsize)  
     trainloader, testloader = dataloader.load()
     # without BN
     base_loss,base_acc = base_model(base_model_dict,trainloader,testloader)
     # With BN
     bn_loss,bn_acc =  bn_model(bn_model_dict,trainloader,testloader)
     #save result
     f = open(opt.net+'_'+opt.dataset+'_result.txt','w')
     f.write(str(base_loss))
     f.write('\n')
     f.write(str(base_acc))
     f.write('\n')
     f.write(str(bn_loss))
     f.write('\n')
     f.write(str(bn_acc))
     f.close()
     # plot 

     step = np.arange(0,len(list(base_loss.values())[0]))
     #plot_loss = np.array(step_loss)[step.tolist()]
     #plot_acc = np.array(step_acc)[step.tolist()]
     plt.figure(figsize = (12,4))
     plt.subplot(121)
     plt.plot(step*10,list(base_loss.values())[0],'k',label = list(base_loss.keys())[0])  # base model without BN
     plt.plot(step*10,list(bn_loss.values())[0],'r:',label = list(bn_loss.keys())[0])  # bn model with lr = opt.lr
     plt.plot(step*10,list(bn_loss.values())[1],'g--',label = list(bn_loss.keys())[1])  # bn model with lr = 5*opt.lr
     plt.plot(step*10,list(bn_loss.values())[2],'b-.',label = list(bn_loss.keys())[2])  # bn model with lr = 30*opt.lr
     plt.xlabel('steps')
     plt.ylabel('loss')
     plt.legend()
     plt.title('test loss')

     plt.subplot(122)
     plt.plot(step*10,list(base_acc.values())[0],'k',label = list(base_loss.keys())[0])  # base model without BN
     plt.plot(step*10,list(bn_acc.values())[0],'r:',label = list(bn_loss.keys())[0])  # bn model with lr = opt.lr
     plt.plot(step*10,list(bn_acc.values())[1],'g--',label = list(bn_loss.keys())[1])  # bn model with lr = 5*opt.lr
     plt.plot(step*10,list(bn_acc.values())[2],'b-.',label = list(bn_loss.keys())[2])  # bn model with lr = 30*opt.lr  
     plt.xlabel('steps')
     plt.ylabel('accuracy')
     plt.legend()
     plt.title('test accuracy')
     plt.savefig(opt.net+'_'+opt.dataset+'_result.png')

