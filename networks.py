import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
from model_utils import get_norm_layer, initialize_weights


def generator(args, gpu_ids=[]):
    norm_layer = get_norm_layer(args.norm_type)
    if args.modelG == 'resnet_9blocks' or args.modelG == 'resnet_9blocks':
        netG = ResnetG(args, gpu_ids=gpu_ids)
    elif args.modelG == 'UnetG':
        netG = UnetG(args, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Enter the proper name of G')
    if len(gpu_ids) > 0:
        netG = netG.cuda(gpu_ids[0])
    netG.apply(initialize_weights)
    return netG


def discriminator(args, gpu_ids=[]):
    norm_layer = get_norm_layer(args.norm_type)
    if args.modelD == 'n_layer':
        netD = NLayerD(args, gpu_ids=gpu_ids, n_layers=3)
    elif args.modelD == 'pixel':
        netD = PixelD(args, gpu_ids)
    else:
        raise NotImplementedError('{} is not an appropriate modelG name'.format(args.modelD))
    if len(gpu_ids) > 0:
        netD = netD.cuda(gpu_ids[0])
    netD.apply(initialize_weights)
    return netD


def scheduler_lr(args, optimizer):
    if args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, 
                                        step_size = args.lr_decay_iters,
                                        gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    else:
        return NotImplementedError('{} is not the proper name of scheduler_lr'.format(args.lr_policy))
    return scheduler


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        padding_size=0
        # padding for first layer of block
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding_size = 1
        else:
            raise NotImplementedError('padding {} is not implemented'.format(padding_type))
        # First layer of convblock
        conv_layer = nn.Conv2d(in_channels = dim,
                               out_channels = dim,
                               kernel_size=3,
                               padding=padding_size,
                               bias=use_bias)
        conv_block += [conv_layer, 
                       norm_layer(dim), 
                       nn.ReLU(True)]
        if use_dropout == True:
            conv_block += [nn.Dropout(0.5)]
        # padding for next layer of block
        padding_size = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding_size = 1
        else:
            raise NotImplementedError('pading {} is not implemented'.format(padding_type))
        # Next layer of convblock
        conv_block += [conv_layer, norm_layer(dim)]
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        out = x + self.conv_block(x)
        return out
    
    
class ResnetG(nn.Module):
    def __init__(self, args, gpu_ids=[]):
        super(ResnetG, self).__init__()
        self.args = args
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(self.args.norm_type)
        # use_bias is dependent on normalizing type
        if norm_layer == nn.InstanceNorm2d:
            use_bias = True
        else:
            use_bias = False
        # Type of Generator
        if args.modelG == 'resnet_9blocks':
            n_blocks=9
        elif args.model_G == 'resnet_6blocks':
            n_blocks=6
        else:
            raise NotImplementedError('{} is not an appropriate modelG name'.format(args.modelG))
        assert(n_blocks>=0)
        #Construct a model
        n_downsampling = 2
        mult = 2 ** n_downsampling
        # First Module
        conv_layer = nn.Conv2d(in_channels = args.ch_in,
                               out_channels = args.ngf,
                               kernel_size=7,
                               padding=0,
                               bias=use_bias)
        layers = [nn.ReflectionPad2d(3), 
                  conv_layer, 
                  norm_layer(args.ngf, affine=True),
                  nn.ReLU(True)]
        # Downsamples with conv layers
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_layer = nn.Conv2d(in_channels = args.ngf * mult,
                                   out_channels = args.ngf * mult * 2,
                                   kernel_size = 3,
                                   stride = 2,
                                   padding = 1,
                                   bias = use_bias)
            layers += [conv_layer,
                       norm_layer(args.ngf * mult * 2, affine=True),
                       nn.ReLU(True)]
        # Construct Resnet blocks
        for i in range(n_blocks):
            resnet_block = ResnetBlock(dim = args.ngf * mult * 2,
                                       padding_type = args.padding_type,
                                       norm_layer = norm_layer,
                                       use_dropout = args.use_dropout, 
                                       use_bias = use_bias)
            layers += [resnet_block]
        # Construct TransposeConv layers
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            trans_layer = nn.ConvTranspose2d(in_channels = args.ngf*mult,
                                             out_channels = int(args.ngf*mult/2),
                                             kernel_size = 3,
                                             stride = 2,
                                             padding = 1,
                                             output_padding = 1,
                                             bias = use_bias)
            layers += [trans_layer,
                      norm_layer(int(args.ngf*mult/2)),
                      nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3)]
        layers += [nn.Conv2d(args.ngf, args.ch_out, kernel_size=7, padding=0)]
        layers += [nn.Tanh()]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, ch_input, ch_inner,
                 submodule=None, outermost=False, innermost=False, norm_layer = nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if norm_layer == nn.InstanceNorm2d:
            use_bias = True
        else:
            use_bias = False
        ### Downsampling Module
        downconv = nn.Conv2d(in_channels = ch_input,
                             out_channels = ch_inner,
                             kernel_size=4,
                             stride=2,
                             padding=1,
                             bias=use_bias)
        downnorm = norm_layer(ch_inner)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(ch_input)
        if innermost==True:
            upconv = nn.ConvTranspose2d(in_channels = ch_inner,
                                        out_channels = ch_input,
                                        kernel_size = 4,
                                        stride = 2,
                                        padding = 1,
                                        bias = use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        ### U-Net's input of transconv = ch_inner*2
        ### Why? torch.cat((prev transconv layer, prev convlayer), 1)
        elif outermost == True:
            upconv = nn.ConvTranspose2d(in_channels = ch_inner*2, 
                                        out_channels = ch_input,
                                        kernel_size = 4,
                                        stride = 2,
                                        padding = 1,
                                        bias = use_bias)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        else:
            upconv = nn.ConvTranspose2d(in_channels = ch_inner*2, 
                                        out_channels = ch_input,
                                        kernel_size = 4,
                                        stride = 2,
                                        padding = 1,
                                        bias = use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout == True:
                model = model + [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        if self.outermost==True:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

        
class UnetG(nn.Module):
    def __init__(self, args, num_downsamples=7, gpu_ids=[]):
        super(UnetG, self).__init__()
        norm_layer = get_norm_layer(args.norm_type)
        # use_bias is dependent on normalizing type
        if norm_layer == nn.InstanceNorm2d:
            use_bias = True
        else:
            use_bias = False
        ### Start from innermost
        unet_block = UnetSkipConnectionBlock(ch_input=args.ngf*8,
                                             ch_inner=args.ngf*8,
                                             submodule=None,
                                             innermost=True,
                                             norm_layer = norm_layer)
        for i in range(num_downsamples - 5):
            unet_block = UnetSkipConnectionBlock(ch_input=args.ngf*8,
                                                 ch_inner=args.ngf*8,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer,
                                                 use_dropout=args.use_dropout)
        unet_block = UnetSkipConnectionBlock(ch_input=args.ngf * 4,
                                             ch_inner=args.ngf * 8,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ch_input=args.ngf * 2,
                                             ch_inner=args.ngf * 4,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ch_input=args.ngf,
                                             ch_inner=args.ngf * 2,
                                             submodule=unet_block, 
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ch_input=args.ch_in,
                                             ch_inner=args.ngf, 
                                             submodule=unet_block, 
                                             outermost=True, 
                                             norm_layer=norm_layer)
        self.model = unet_block
        
    def forward(self, x):
        return self.model(x)

        
class PixelD(nn.Module):
    def __init__(self, args, gpu_ids=[]):
        super(PixelD, self).__init__()
        self.args = args
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(self.args.norm_type)
        # use_bias is dependent on normalizing type
        if norm_layer == nn.InstanceNorm2d:
            use_bias = True
        else:
            use_bias = False
        conv_layer1 = nn.Conv2d(in_channels = self.ch_in,
                                out_channels = self.ndf,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        conv_layer2 = nn.Conv2d(in_channels = self.ndf,
                                out_channels = self.ndf*2,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias = use_bias)
        conv_layer3 = nn.Conv2d(in_channels = self.ndf*2,
                                out_channels = 1,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=use_bias)
        layers = [conv_layer1, nn.LeakyReLU(0.2, True),
                 conv_layer2, norm_layer(args.ndf*2), nn.LeakyReLU(0.2, True),
                 conv_layer3]
        if args.use_sigmoid==True:
            layers += [nn.Sigmoid()]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    
class NLayerD(nn.Module):
    def __init__(self, args, gpu_ids=[], n_layers=3):
        super(NLayerD, self).__init__()
        self.args = args
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(self.args.norm_type)
        # use_bias is dependent on normalizing type
        if norm_layer == nn.InstanceNorm2d:
            use_bias = True
        else:
            use_bias = False
            
        conv_layer = nn.Conv2d(in_channels = args.ch_in,
                               out_channels = args.ndf, 
                               kernel_size = 4,
                               stride=2,
                               padding=1)
        layers = [conv_layer, nn.LeakyReLU(0.2, True)]
        
        nf_mult_prev = 1
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            conv_layer = nn.Conv2d(in_channels = args.ndf * nf_mult_prev,
                                   out_channels = args.ndf * nf_mult,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=use_bias)
            layers += [conv_layer, norm_layer(args.ndf*nf_mult), nn.LeakyReLU(0.2,True)]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        conv_layer = nn.Conv2d(in_channels = args.ndf * nf_mult_prev,
                               out_channels = args.ndf * nf_mult,
                               kernel_size=4,
                               stride=1,
                               padding=1,
                               bias=use_bias)
        final_layer = nn.Conv2d(in_channels = args.ndf * nf_mult,
                               out_channels = 1,
                               kernel_size=4,
                               stride=1,
                               padding=1)
        layers += [conv_layer, norm_layer(args.ndf * nf_mult), nn.LeakyReLU(0.2, True), final_layer]
        if args.use_sigmoid==True:
            layers += [nn.Sigmoid()]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
