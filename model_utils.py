import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    # elif classname.find('InstanceNorm2d') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)
        
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm_type == 'None':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
class GANLoss(nn.Module):
    def __init__(self, is_lsgan, is_real_label, is_cuda=True):
        super(GANLoss, self).__init__()
        self.is_real_label = is_real_label
        self.is_cuda = is_cuda
        if is_lsgan == True:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
    
    def target_tensor(self, _input):
        if self.is_real_label == True:
            targets = Variable(torch.ones(_input.size()), requires_grad=False)
        else:
            targets = Variable(torch.zeros(_input.size()), requires_grad=False)
        
        if self.is_cuda==True:
            targets = targets.cuda()
        return targets
    
    def __call__(self, _input):
        targets = self.target_tensor(_input)
        
        return self.loss(_input, targets)