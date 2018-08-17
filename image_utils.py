import os
import numpy as np
import torch
from PIL import Image

def is_image(file):
    return any(file.endswith(extension) for extension in ['.jpg','.png','.jpeg'])

def tensor2img(img_tensor, img_type=np.uint8):
    if isinstance(img_tensor, torch.Tensor):
        img_data = img_tensor.data
    np_img = img_data[0].cpu().float().numpy()
    if np_img.shape[0] == 1:
        np_img = np.tile(np_img, (3, 1, 1))
    np_img = 255.0 * (np.transpose(np_img, (1,2,0)) + 1) / 2.0
    np_img = np_img.astype(img_type)
    return np_img

def load_img(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((256,256), Image.BICUBIC)
    return img

def save_image(data, path):
    image = Image.fromarray(data)
    image.save(path)
    
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)