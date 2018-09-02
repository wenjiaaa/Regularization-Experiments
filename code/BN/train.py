#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:27:32 2018

@author: wenjia
"""
import numpy as np
import torch
from torchvision.datasets import  MNIST
from torch.autograd import Variable

class Trainer(object):
    def __init__(self, device,model, criterion):
        super(Trainer, self).__init__()
        self.device = device  # GPU or CPU
        self.model = model
        self.criterion = criterion
 
    def test(self,testloader):
        #self.model.test()
        sum = 0
        runloss = 0
        correct = 0
        for _, batch in enumerate(testloader):
            input, target = batch
            input,target = input.to(self.device),target.to(self.device)
            sum += len(target)
            input = Variable(input)
            target = Variable(target)
            output = self.model(input)
            loss = self.criterion(output, target)
            runloss += loss.data[0]
            _, predict = torch.max(output, 1)
            correctnum = (predict == target).sum()
            correct += correctnum.data[0]
 
        epoch_loss = float(runloss) / sum
        poch_correct = int(correct) / sum
        return epoch_loss,poch_correct


    def train(self,epoch,trainloader,testloader,optimizer):
        #self.model.train()
        step_loss = []
        step_acc = []
        for i in range(epoch):
            sum=0
            runloss=0
            correct=0
            for step,batch in enumerate(trainloader):
 
                input,target=batch
                input,target = input.to(self.device),target.to(self.device)
                sum+=len(target)
 
                input=Variable(input)
                target=Variable(target)
                output=self.model(input)
                loss=self.criterion(output,target)
                optimizer.zero_grad()
 
                loss.backward()
                optimizer.step()
                runloss+=loss.data[0]
                _,predict=torch.max(output,1)
                #predict  = predict + 1
                correctnum=(predict==target).sum()
                correct+=correctnum.data[0]
                #if i == 0 and step % 10 ==0:
                #if step % 500 == 0 :
                    
                 #   test_loss,test_acc = self.test(testloader)
                  #  print(test_loss,test_acc)
                   # step_loss.append(test_loss)
                    #step_acc.append(test_acc)
            epoch_loss=runloss/sum
            epoch_correct=int(correct)/sum
            #print(correct.data[0])
            print("epoch {:d}  |epoch loss {:f}   |epoch_correct {:f}".format(i,epoch_loss,epoch_correct))
        return step_loss,step_acc
        

        
              
