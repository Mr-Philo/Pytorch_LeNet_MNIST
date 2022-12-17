import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model.lenet import LeNet
from data.build_data import dataloader

try:
    import wandb
except ImportError:
    wandb = None

def get_parse():
    parser = argparse.ArgumentParser('Pytorch_LeNet_MNIST', add_help=False)
    parser.add_argument('--data-path', type=str, default='./downloads/', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=128, help="batch size for single GPU")
    parser.add_argument('--epoch', type=int, default=10, help="set epoches for training")
    parser.add_argument('--lr', type=float, default=0.001, help="set learning rate for training")
    parser.add_argument('--optim', type=str, default='Adam', help='set optimizer for training')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--use-wandb', action='store_true', default=False, help='use wandb to record log')
    
    args = parser.parse_args()
    return args

def main(args):
    
    # prepare dataset
    print("Dataset folder: '{}'".format(args.data_path))
    train_loader, test_loader = dataloader(args.data_path, args.batch_size)
    
    # prepare model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = LeNet().to(device)
    
    # define loss function
    loss_f = nn.CrossEntropyLoss()
    
    # define optimizer
    optimizer = optim.Adam(net.parameters(), lr = args.lr)
    
    # start training
    print("Start training, total epoches: {}".format(args.epoch))
    start = time.time()
    for epoch in range(args.epoch):
        
        loss_sum,acc_sum,total_num,total_step = 0,0,0,0
        epoch_start = time.time()
        net.train()
        
        # train each iter in dataloader
        for step,(img,label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)
            
            out = net(img)          # forward
            loss = loss_f(out, label)    # compute loss
            loss_sum += loss.item()
            
            optimizer.zero_grad()
            loss.backward()         # back propogation
            optimizer.step()        # update weight of the model
            
            prediction = torch.argmax(out, dim=1)
            acc = (prediction == label).sum().item()    # compute acc
            acc_sum += acc
            total_num += len(label)
            total_step += 1
            
            # print("Step[{}] in epoch[{}/{}]: loss {:.3f}, accuracy {:.3f}".format(
            #     step, epoch+1, args.epoch, loss.item(), acc/len(label)
            # ))
            
        epoch_end = time.time()
        print("Epoch[{}/{}]: loss {:.3f}, accuracy {:.3f}, time cost {:.3f}s".format(
                epoch+1, args.epoch, loss_sum/total_step, acc_sum/total_num, epoch_end-epoch_start
            ))
        
        # Validate
        # TODO
    
    if not os.path.exists("./checkpoints/"):
        os.makedirs("./checkpoints/")
    torch.save(net, "./checkpoints/model.pth")
    end = time.time()
    print("Training finished, total time cost {:.3f}min".format((end-start)/60))
    

if __name__ == '__main__':
    args = get_parse()
    main(args)
