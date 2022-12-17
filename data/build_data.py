import torch
import torchvision
import torch.utils.data as Data
import os

def dataloader(path, batchsize):
    
    DOWNLOAD = False
    if not(os.path.exists(path)) or not os.listdir(path):
    # not mnist dir or mnist is empyt dir
        DOWNLOAD = True
    else:
        print("MNIST dataset already exist in '{}', skip download".format(path))

    train_data = torchvision.datasets.MNIST(
        root = path,
        train = True,
        transform = torchvision.transforms.ToTensor(),
        download = DOWNLOAD
    )
    
    test_data = torchvision.datasets.MNIST(
        root = path,
        train = False,
        transform = torchvision.transforms.ToTensor(),
        download = DOWNLOAD
    )
    
    train_loader = Data.DataLoader(
        dataset = train_data,
        batch_size = batchsize,
        shuffle = True
    )
    
    test_loader = Data.DataLoader(
        dataset = test_data,
        batch_size = batchsize
    )
    
    return train_loader, test_loader

if __name__ == '__main__':
    path = '../downloads/'
    train_loader, test_loader = dataloader(path, 128)
    print('done')