import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from model_utils import *

def define_G(ch_in, ch_out, ngf, model_type, norm_type='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type)
    if model_type == 'resnet_9blocks':
        netG = ResnetG(ch_in, ch_out, ngf, norm_layer, use_dropout, 9)
    elif model_type == 'resnet_6blocks':
        netG = ResnetG(ch_in, ch_out, ngf, norm_layer, use_dropout, 6)
    else:
        raise NotImplementedError('Enter the proper name of G')
    if len(gpu_ids) > 0:
        netG = netG.cuda(gpu_ids[0])
    netG.apply(initialize_weights)
    return netG

def define_D(ch_in, ndf, model_type, n_layers=3, norm_type='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type)
    if model_type == 'n_layer':
        netD = NLayerD(ch_in, ndf, n_layers, norm_layer, use_sigmoid, gpu_ids)
    elif model_type == 'pixel':
        netD = PixelD(ch_in, ndf, norm_layer, use_sigmoid, gpu_ids)
    else:
        raise NotImplementedError('Enter the proper name of D')
    if len(gpu_ids) > 0:
        netD = netD.cuda(gpu_ids[0])
    netD.apply(initialize_weights)
    return netD


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        
        padding_size=0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding_size = 1
        else:
            raise NotImplementedError('pading {} is not implemented'.format(padding_type))
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=padding_size, bias=use_bias), 
                       norm_layer(dim), nn.ReLU(True)]
        if use_dropout == True:
            conv_block += [nn.Dropout(0.5)]
        
        padding_size = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding_size = 1
        else:
            raise NotImplementedError('pading {} is not implemented'.format(padding_type))
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=padding_size, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        out = x+ self.conv_block(x)
        return out
    
    
class ResnetG(nn.Module):
    def __init__(self, ch_in, ch_out, ngf = 64, norm_layer = nn.BatchNorm2d, use_dropout=False, n_blocks = 6, padding_type = 'reflect', gpu_ids = []):
        assert(n_blocks>=0)
        super(ResnetG, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if norm_layer == nn.InstanceNorm2d:
            use_bias = True
        else:
            use_bias = False
        model = [nn.ReflectionPad2d(3), nn.Conv2d(ch_in, ngf, kernel_size=7, padding=0,bias=use_bias), norm_layer(ngf, affine=True), nn.ReLU(True)]
        
        n_downsampling=2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf*mult*2, affine=True), nn.ReLU(True)]
        
        mult = 2** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf*mult, padding_type, norm_layer, use_dropout, use_bias)]
        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                     norm_layer(int(ngf*mult/2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, ch_out, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    
    
# PatchGAN D
class NLayerD(nn.Module):
    def __init__(self, ch_in, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerD, self).__init__()
        self.gpu_ids = gpu_ids
        if norm_layer == nn.InstanceNorm2d:
            use_bias = True
        else:
            use_bias = False
        kernel_size, padding_size = 4, 1
        model = [nn.Conv2d(ch_in, ndf, kernel_size, 2, padding_size), nn.LeakyReLU(0.2, True)]
        
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, n_layers):
            nf_mult_prev=nf_mult
            nf_mult = min(2**n, 8)
            model += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size, 2, padding_size, bias=use_bias), 
                     norm_layer(ndf*nf_mult), nn.LeakyReLU(0.2,True)]
        
        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        model += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size, 1, padding_size, bias=use_bias), 
                 norm_layer(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        model += [nn.Conv2d(ndf*nf_mult, 1, kernel_size, 1, padding_size)]
        
        if use_sigmoid:
            model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    
    
class PixelD(nn.Module):
    def __init__(self, ch_in, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelD, self).__init__()
        self.gpu_ids = gpu_ids
        if norm_layer == nn.InstanceNorm2d:
            use_bias = True
        else:
            use_bias = False
        self.network = [
            nn.Conv2d(ch_in, ndf, 1, 1, 0), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 1, 1, 0, bias=use_bias), norm_layer(ndf*2), nn.LeakyReLU(0.2, True), 
            nn.Conv2d(ndf*2, 1, 1, 1, 0, bias=use_bias)
        ]
        if use_sigmoid:
            self.network += [nn.Sigmoid()]
        self.model = nn.Sequential(*self.network)
        
    def forward(self, x):
        return self.model(x)