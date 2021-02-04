import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets
import torchvision.transforms as transforms
import argparse
import numpy as np
from train import train
from models.VT import VT
import sys
import os
import random

def parse_args():
    parser = argparse.ArgumentParser()

    #data parameters
    parser.add_argument('-i', '--image_size', help='(square) dimension of 2D input image', type=int)
    parser.add_argument('-p', '--patch_size', help='(square) dimension of 2D image patches', type=int)
    parser.add_argument('-ch', '--channels', help='image channels', type=int)
    parser.add_argument('-cl', '--num_classes', help='number of class labels', type=int)

    #model parameters
    parser.add_argument('-d', '--dim', help='internal model dimension', type=int, default=128)
    parser.add_argument('-de', '--depth', help='model depth', type=int, default=4)
    parser.add_argument('-he', '--heads', help='model heads', type=int, default=8)
    parser.add_argument('-dh', '--dim_head', help='model heads dimension', type=int, default=64)
    parser.add_argument('-md', '--mlp_dim', help='MLP dimension', type=int, default=64)
    parser.add_argument('-po', '--pool', help='pooling type;\"cls\" or \"mean\"', type=str, default='cls')
    parser.add_argument('-do', '--dropout', help='model droput rate', type=float, default=0.)
    parser.add_argument('-edo', '--emb_dropout', help='embedding droput rate', type=float, default=0.)

    #training parameters
    parser.add_argument('-b', '--batch_size', help='batch size for training', type=int, default=128)
    parser.add_argument('-e', '--epochs', help='number of epochs to train', type=int, default=20)
    parser.add_argument('-l', '--learning_rate', help='learning rate', type=float, default=3E-5)
    parser.add_argument('-ga', '--gamma', help='learning rate decay rate', type=float, default=0.7)
    parser.add_argument('-g', '--gpu', help='GPU ID', type=str, default=0)
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=42)


    args = parser.parse_args()

    return args


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.device(device)

    transform_train = transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ])

    transform_val = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                       ])

    train_set    = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_set    = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=True, num_workers=2)

    model = VT(args).to(device)
    
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    seed_everything(args.seed)


    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    model = VT(args)

    train(model, train_loader, val_loader, device, criterion, optimizer, args.epochs)



if __name__ == '__main__':

    args = parse_args()

    main(args)