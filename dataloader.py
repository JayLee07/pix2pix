import os
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader
from image_utils import is_image, load_img

class ImageLoader(data.Dataset):
    def __init__(self, filepath, data_type='train'):
        super(ImageLoader, self).__init__()
        self.a_path = os.path.join(filepath, 'a', data_type)
        self.b_path = os.path.join(filepath, 'b', data_type)
        self.files = [x for x in os.listdir(self.a_path) if is_image(x)]
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
    
    def __getitem__(self, index):
        _input = load_img(os.path.join(self.a_path, self.files[index]))
        _input = self.transform(_input)
        _target = load_img(os.path.join(self.b_path, self.files[index]))
        _target = self.transform(_target)
        return _input, _target
    
    def __len__(self):
        return len(self.files)
    
def get_train_set(filepath):
    return ImageLoader(filepath, data_type='train')
def get_test_set(filepath):
    return ImageLoader(filepath, data_type='test')