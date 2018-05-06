#!/usr/bin/env python3

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class BasicConv(nn.Module):
    def __init__(self):
        super(BasicConv, self).__init__()

