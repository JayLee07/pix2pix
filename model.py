import os, sys, time, pickle
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.autograd import Variable
from model_utils import *
from image_utils import tensor2img, save_image
from networks import *
from dataloader import *

# Loss function definition
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()
L1_loss = nn.L1Loss()
real_bce_gan_loss = GANLoss(False, True, True)
fake_bce_gan_loss = GANLoss(False, False, True)
real_ls_gan_loss = GANLoss(True, True, True)
fake_ls_gan_loss = GANLoss(True, False, True)

class Pix2Pix(object):
    def __init__(self, args, trn_loader, tst_loader, sample_a, sample_b):
        self.args = args
        self.trn_loader = trn_loader
        self.tst_loader = tst_loader
        self.G = generator(args, gpu_ids = args.gpu_ids)
        self.D = discriminator(args, gpu_ids=args.gpu_ids)
        self.optimD = torch.optim.Adam(params=self.D.parameters(), lr = args.lrD)
        self.optimG = torch.optim.Adam(params=self.G.parameters(), lr = args.lrG)
        self.sample_a = sample_a
        self.sample_b = sample_b
        if len(args.gpu_ids) > 0:
            self.is_cuda=True
        else:
            self.is_cuda=False
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def trainD(self, real_a, real_b, fake_b):
        """
        Train the discriminator
        """
        self.set_requires_grad(self.D, True)
        self.set_requires_grad(self.G, False)
        self.optimD.zero_grad()
        ### Train with fake pair
        fake_ab = torch.cat((real_a, fake_b), 2)
        D_fake_ab = self.D.forward(fake_ab.detach())
        D_fake_loss = fake_bce_gan_loss(D_fake_ab)
        ### Train with real pair
        real_ab = torch.cat((real_a, real_b), 2)
        D_real_ab = self.D.forward(real_ab.detach())
        D_real_loss = real_bce_gan_loss(D_real_ab)
        self.loss_dict['D_fake_loss'] = D_fake_loss
        self.loss_dict['D_real_loss'] = D_real_loss
        return D_fake_loss, D_real_loss
    
    
    def trainG(self, real_a, real_b, fake_b):
        """
        Train the generator
        """
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, True)
        self.optimG.zero_grad()
        ### Train with fake pair.
        fake_ab = torch.cat((real_a, fake_b), 2)
        D_fake_ab = self.D.forward(fake_ab)
        G_fake_loss = real_ls_gan_loss(D_fake_ab)        
        ### B=G(A)
        l1_loss = L1_loss(fake_b, real_b) * self.args.lamb
        G_loss = G_fake_loss + l1_loss
        self.loss_dict['G_loss'] = G_loss
        return G_loss
    
    
    def train(self):
        """
        Train the whole model
        """
        self.loss_dict = dict()
        self.loss_dict['G_loss'] = list()
        self.loss_dict['D_fake_loss'] = list()
        self.loss_dict['D_real_loss'] = list()
        print('>>> Start Training')
        for epoch in range(self.args.maxepoch):
            self.set_requires_grad(self.D, True)
            self.set_requires_grad(self.G, True)
            start = time.time()
            for iter_num, batch in enumerate(self.trn_loader):
                print("\r>>>>>>Epoch {}, iter_num {}/400".format(epoch+1, iter_num+1), end="")
                real_a, real_b = batch[0], batch[1]
                if self.is_cuda == True:
                    real_a = real_a.cuda()
                    real_b = real_b.cuda()
                fake_b = self.G.forward(real_a)
                # Train D
                D_fake_loss, D_real_loss = self.trainD(real_a, real_b, fake_b)
                D_loss = 0.5 * (D_fake_loss + D_real_loss)
                D_loss.backward()
                self.optimD.step()
                # Train G
                G_loss = self.trainG(real_a, real_b, fake_b)
                G_loss.backward()
                self.optimG.step()
                
                
            print("\nIn epoch: {}, D_fake_loss: {:.4f}, D_real_loss: {:.4f}, G_loss: {:.4f}".format(epoch+1,
                                                                                                  D_fake_loss,
                                                                                                  D_real_loss,
                                                                                                  G_loss))
            self.save_results(epoch, self.sample_a, self.sample_b)
        self.save_model()
    
    def save_results(self, epoch, sample_a, sample_b):
        ### save result img file
        exp_config = "ngf_{}_ndf_{}_lambda_{}_norm_{}".format(self.args.ngf,
                                                              self.args.ndf,
                                                              self.args.lamb,
                                                              self.args.norm_type)
        result_dir = os.path.join(self.args.save_dir,'logs', self.args.model, self.args.dataset, exp_config)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        fake_filename = os.path.join(result_dir, 'epoch%03d.png' %(epoch+1))
        
        # self.G.eval()
        self.set_requires_grad(self.G, False)
        if self.is_cuda == True:
            sample_a = sample_a.cuda()
            sample_b = sample_b.cuda()
        fake_b = self.G.forward(sample_a)
        results = torch.cat((sample_a, fake_b, sample_b), 3)
        results_img = tensor2img(results)
        save_image(results_img, fake_filename)
        
    def save_model(self):
        #save trained models
        exp_config = "ngf_{}_ndf_{}_lambda_{}_norm_{}".format(self.args.ngf,
                                                              self.args.ndf,
                                                              self.args.lamb,
                                                              self.args.norm_type)
        model_dir = os.path.join(self.args.save_dir, self.args.model, self.args.dataset, exp_config)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        torch.save(self.G.state_dict(), os.path.join(model_dir, 'G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(model_dir, 'D.pkl'))
        with open(os.path.join(model_dir, 'loss_dict'), 'wb') as f:
            pickle.dump(self.loss_dict, f)
            
    def load_model(self):
        # load saved model
        exp_config = "ngf_{}_ndf_{}_lambda_{}_norm_{}".format(self.args.ngf,
                                                              self.args.ndf,
                                                              self.args.lamb,
                                                              self.args.norm_type)
        model_dir = os.path.join(self.args.save_dir, self.args.model, self.args.dataset, exp_config)
        
        self.G.load_state_dict(torch.load(os.path.join(model_dir, 'G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(model_dir, 'D.pkl')))