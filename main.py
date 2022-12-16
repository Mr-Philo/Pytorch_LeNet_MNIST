import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
except ImportError:
    wandb = None

def get_parse():
    parser = argparse.ArgumentParser('Pytorch_LeNet_MNIST', add_help=False)
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=16, help="batch size for single GPU")
    parser.add_argument('--epoch', type=int, default=90, help="set epoches for training")
    parser.add_argument('--lr', type=float, default=0.01, help="set learning rate for training")
    parser.add_argument('--optim', type=str, default='Adam', help='set optimizer for training')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--use-wandb', action='store_true', default=False, help='use wandb to record log')
    
    args = parser.parse_args()
    return args

def main(args):
    pass

if __name__ == '__main__':
    args = get_parse()
    main(args)
    