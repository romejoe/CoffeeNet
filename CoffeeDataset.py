from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Compose
from torchvision.transforms.functional import crop


class CoffeeDataset(Dataset):
    """Dataset containing coffee data"""

    def __init__(self, csv_path, root_dir, transform=None):
        self.csv_data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        #Read image and amount
        img_path = os.path.join(self.root_dir, self.csv_data['path'][0])
        raw_img = Image.open(img_path)
        amount = self.csv_data['path'][1]

        #Remove the top label from the image
        image = crop(raw_img,  18,0, raw_img.size[1] - 18, raw_img.size[0])
        tmp = [

        ]

        if self.transform is not None:
            tmp.append(self.transform)

        tmp.append(transforms.ToTensor())

        t = Compose(tmp)

        return [t(image), image, amount]
