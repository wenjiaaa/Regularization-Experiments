#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:08:01 2018

@author: wenjia
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
 
class LeNet(nn.Module):
    def __init__(self,BN = True,channels = 1, num_classes = 10 ):
        super(LeNet, self).__init__()
        self.BN = BN
        if BN == True:
               
               self.conv = nn.Sequential(
               #### 1th conv
               nn.Conv2d(channels, 6, 5),
               nn.BatchNorm2d(6),
               nn.MaxPool2d(2, 2),
               #### 2th conv
               nn.Conv2d(6, 16, 5),
               nn.BatchNorm2d(16),
               nn.MaxPool2d(2, 2))
               
 
               self.fc = nn.Sequential(
               nn.Linear(400, 120),
               nn.BatchNorm1d(120),
               
               nn.Linear(120, 84),
               nn.BatchNorm1d(84),
               
               nn.Linear(84, num_classes))
               
        else:
              
               self.conv = nn.Sequential(
               #### 1th conv
               nn.Conv2d(channels, 12, 5),
               nn.MaxPool2d(2, 2),
               #### 2th conv
               nn.Conv2d(12, 32, 5),
               nn.MaxPool2d(2, 2))
 
               self.fc = nn.Sequential(
               nn.Linear(800, 240),
               nn.Linear(240,240),
               nn.Linear(240, num_classes))
 
 
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
 
        out = self.fc(out)
        return out
