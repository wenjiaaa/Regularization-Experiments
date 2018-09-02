#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 17:45:21 2018

@author: wenjia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:27:32 2018

@author: wenjia
"""
# import numpy as np
import torch
# from torchvision.datasets import  MNIST
from torch.autograd import Variable

class Trainer(object):
    def __init__(self, device ,model, criterion,reg_type):
        super(Trainer, self).__init__()
        self.device = device  # GPU or CPU
        self.model = model
        self.criterion = criterion
        self.reg_type = reg_type

    def calculate_reg_loss(self, model ,loss,reg_type,mu):
        #mu = torch.tensor(mu)
        reg = torch.tensor(0.)
        reg = reg.to(self.device)
        if  reg_type == 'L1':
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    reg += torch.norm(param,1)
        elif reg_type == 'L2' :
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    reg += torch.norm(param,2)*torch.norm(param,2)

        reg_loss = loss + mu * reg
        return reg_loss
                
    def test(self,testloader):
        #self.model.test()
        sum = 0
        runloss = 0
        correct = 0
        for _, batch in enumerate(testloader):
            input, target = batch
            # input = input.reshape(-1, 28*28)   #28*28 -->1*784
            input,target = input.to(self.device),target.to(self.device)
            sum += len(target)
            input = Variable(input)
            target = Variable(target)
            output = self.model(input
                                )
            loss = self.criterion(output, target)
            runloss += loss.data[0]
            _, predict = torch.max(output, 1)
            correctnum = (predict == target).sum()
            correct += correctnum.data[0]
 
        epoch_loss = float(runloss) / sum
        epoch_correct = int(correct) / sum
        return epoch_loss,epoch_correct


    def train(self,epoch,trainloader,testloader,optimizer):
        #self.model.train()
        test_loss = []
        test_acc = []
        for i in range(epoch):
            sum=0
            runloss=0
            correct=0
            for step,batch in enumerate(trainloader):
 
                input,target=batch
                # input = input.reshape(-1, 28*28)   #28*28 -->1*784

                input,target = input.to(self.device),target.to(self.device)
                sum+=len(target)
 
                input=Variable(input)
                target=Variable(target)
                output=self.model(input)
                loss=self.criterion(output,target)
                #print(loss)
                reg_loss = self.calculate_reg_loss(self.model,loss,self.reg_type,1e-3)
                optimizer.zero_grad()
 
                reg_loss.backward()
                optimizer.step()

                runloss+=loss.data[0]
                _,predict=torch.max(output,1)
                #predict  = predict + 1
                correctnum=(predict==target).sum()
                correct+=correctnum.data[0]

                if i <= epoch and step % 50 ==0:
                #if step % 500 == 0 :
                    test_loss_i,test_acc_i = self.test(testloader)
                    test_loss.append(test_loss_i)
                    test_acc.append(test_acc_i)

            epoch_train_loss = runloss/sum
            epoch_train_acc=int(correct)/sum
            epoch_test_loss,epoch_test_acc = self.test(testloader)
            #print(correct.data[0])
            print("epoch {:d}  |train loss {:f}   |train acc {:f}"
                  '\n'
                  '            |test loss {:f}   test acc  {:f}'.format(i,epoch_train_loss,epoch_train_acc,epoch_test_loss,epoch_test_acc))
        test_acc.append(epoch_test_acc)
        test_loss.append(epoch_test_loss)
        return test_loss,test_acc
        

        
              
