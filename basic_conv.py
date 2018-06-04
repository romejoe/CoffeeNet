#!/usr/bin/env python3

from __future__ import print_function

import argparse

import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from torchvision.transforms import Resize
from CoffeeDataset import CoffeeDataset

parser = argparse.ArgumentParser()

parser.add_argument("--data_csv", type=str)
parser.add_argument("--data_root", type=str, default='data')

config = parser.parse_args()

ds = CoffeeDataset('./data/labels.csv', './data/', transform=Resize(224))

tmp = ds[0]

#tmp[1].show()


class BasicConv(nn.Module):
    def __init__(self):
        super(BasicConv, self).__init__()
        self.conv1 = nn.Conv2d(3 * 224, 64, kernel_size=5, stride=2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.fc(x)

        return x


net = BasicConv()
print(net)

print(net.forward(tmp[0]))
