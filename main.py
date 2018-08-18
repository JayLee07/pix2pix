import os, sys, time, pickle
import parser
import model
import argparse
from dataloader import get_train_set, get_test_set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def argument():
    parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
    parser.add_argument('--dataset', required=True, default='facades')
    parser.add_argument('--trainBatchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--maxepoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--ch_in', type=int, default=3, help='input image channels')
    parser.add_argument('--ch_out', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters')
    parser.add_argument('--lrD', type=float, default=0.0002, help='Learning Rate D')
    parser.add_argument('--lrG', type=float, default=0.0002, help='Learning Rate G')
    parser.add_argument('--lr_policy', type=str, default='step', help='learning rate schedule')
    parser.add_argument('--lr_decay_iters', type=int, default=30, help='lr decay frequency')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--model', type=str, default='pix2pix')
    parser.add_argument('--modelD', type=str, default='n_layer')
    parser.add_argument('--modelG', type=str, default='resnet_9blocks')
    parser.add_argument('--padding_type', type=str, default='reflect')
    parser.add_argument('--norm_type', type=str, default='instance')
    parser.add_argument('--use_dropout', type=bool, default='False')
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--use_sigmoid', type=bool, default='True')
    parser.add_argument('--gpu_ids', type=int, nargs='+', help='# of gpu_ids')

    opt = parser.parse_args()
    return opt

args = argument()

# Dataset and Dataloader Definition
trn_dset = get_train_set('/data/jehyuk/imgdata/datasets/facades/')
tst_dset = get_test_set('/data/jehyuk/imgdata/datasets/facades/')
sample_a = tst_dset[0][0].unsqueeze(dim=0)
sample_b = tst_dset[0][1].unsqueeze(dim=0)
trn_loader = DataLoader(dataset=trn_dset, num_workers=2, batch_size=1, shuffle=True)
tst_loader = DataLoader(dataset=tst_dset, num_workers=2, batch_size=1, shuffle=False)

pix2pix = model.Pix2Pix(args, trn_loader, tst_loader, sample_a, sample_b)
pix2pix.train()