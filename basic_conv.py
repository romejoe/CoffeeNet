#!/usr/bin/env python3

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import argparse

from CoffeeDataset import CoffeeDataset

parser = argparse.ArgumentParser()

parser.add_argument("--data_csv", type=str)
parser.add_argument("--data_root", type=str, default='data')

config = parser.parse_args()


ds = CoffeeDataset('./data/labels.csv', './data/')

tmp = ds[0]

tmp[1].show()

class BasicConv(nn.Module):
    def __init__(self):
        super(BasicConv, self).__init__()
