#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:19:54 2018

@author: wenjia
"""
import torch	
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, Fc_BN = False, Conv_BN = False , drop_prob = 0,channels=1,num_classes=10):
        super(VGG, self).__init__()
        self.channels = channels
        # self.Fc_BN = Fc_BN
        self.features = self._make_layers(cfg[vgg_name],Conv_BN)
        if Fc_BN:
            self.classifier = nn.Sequential(
                nn.Linear(512,128),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Linear(128, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512,128),
                nn.Dropout(drop_prob),
                nn.ReLU(True),
                nn.Linear(128,128),
                nn.Dropout(drop_prob),
                nn.ReLU(True),
                nn.Linear(128, num_classes)
            ) 
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg ,Conv_BN = False):
        layers = []
        in_channels = self.channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels,x,kernel_size = 3,padding = 1)
                if Conv_BN:
                       layers += [conv2d,nn.BatchNorm2d(x),nn.ReLU(inplace=True)]
                else:
                       layers += [conv2d,nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
