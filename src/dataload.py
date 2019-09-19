import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_size, flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)
        self.image_count = len(self.data)
        self.mask_count = len(self.mask_data)
        print('Images: {}. Masks: {}'.format(self.image_count, self.mask_count))

        self.input_size = input_size
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size
        
        # load image
        img = imread(self.data[index])
        mask = 255 - imread(self.mask_data[torch.randint(high=self.mask_count, size=[1]).int().item()])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        img = scipy.misc.imresize(img, [256, 256], interp='bilinear')
        mask = scipy.misc.imresize(mask, [256, 256], interp='nearest')
        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        if self.augment and np.random.binomial(1, 0.5) > 0:
            mask = mask[::-1, :, ...]
        return self.to_tensor(img), self.to_tensor(mask)


    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t


    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        
        return []