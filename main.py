import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets
from data import build_oasis, build_mprage
from transform import Resize, RandomZoom, RandomHorizontalFlip
import torchvision.transforms
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
    parser.add_argument('-da', '--data_set', help='Which data set to train on', type=str, default='cifar10')
    parser.add_argument('-i', '--image_size', help='(square) dimension of 2D input image', type=int)
    parser.add_argument('-p', '--patch_size', help='(square) dimension of 2D image patches', type=int)
    parser.add_argument('-ch', '--channels', help='image channels', type=int)
    parser.add_argument('-cl', '--classes', help='number of class labels', type=int)
    parser.add_argument('-seg', '--segmentation', help='Labels are segmentations', type=bool, default=False)


    #model parameters
    parser.add_argument('-d', '--dim', help='internal model dimension', type=int, default=128)
    parser.add_argument('-de', '--depth', help='model depth', type=int, default=4)
    parser.add_argument('-he', '--heads', help='model heads', type=int, default=8)
    parser.add_argument('-dh', '--dim_head', help='model heads dimension', type=int, default=64)
    parser.add_argument('-md', '--mlp_dim', help='MLP dimension', type=int, default=64)
    parser.add_argument('-po', '--pool', help='pooling type;\"cls\" or \"mean\"', type=str, default='cls')
    parser.add_argument('-do', '--dropout', help='model droput rate', type=float, default=0.)
    parser.add_argument('-edo', '--emb_dropout', help='embedding droput rate', type=float, default=0.)
    parser.add_argument('-n', '--name', help='model name for saving', type=str, default='ViT')

    #training parameters
    parser.add_argument('-b', '--batch_size', help='batch size for training', type=int, default=128)
    parser.add_argument('-e', '--epochs', help='number of epochs to train', type=int, default=20)
    parser.add_argument('-l', '--learning_rate', help='learning rate', type=float, default=3E-3)
    parser.add_argument('-ga', '--gamma', help='learning rate decay rate', type=float, default=0.7)
    parser.add_argument('-g', '--gpu', help='GPU ID', type=str, default='0')
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=42)
    parser.add_argument('-a', '--aug', help='toggle data augmentation', type=bool, default=True)
    parser.add_argument('--patience', help='Early stopping patience', type=int, default=100)


    args = parser.parse_args()

    return args

def get_weights(set, reps=20):

    w=0
    for _ in range(reps):
        rand_sample = set[np.random.randint(len(set))][1]
        occ = np.array([np.sum(rand_sample == i) for i in np.unique(rand_sample)])
        w += 1 - occ / np.sum(occ)
    
    return np.asarray(w) / reps

def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pre_train_mean, pre_train_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    if args.channels == 1:
        pre_train_mean = np.mean(pre_train_mean)
        pre_train_std = np.mean(pre_train_std)

    transform_train = transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                        #   transforms.RandomAffine(degrees=(-15,15), translate=(.25,.25), scale=(.1,.3), shear=(-5,5)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(pre_train_mean, pre_train_std),
                                         ])

    transform_val = transforms.Compose([
                                        transforms.Resize(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize(pre_train_mean, pre_train_std),
                                       ])


    if args.data_set.lower() == 'cifar10':
        train_set    = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_set      = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        val_loader   = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=True, num_workers=2)
    elif args.data_set.lower() == 'mnist':
        train_set    = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_set      = datasets.MNIST(root='./data', train=False, download=True, transform=transform_val)
        val_loader   = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=True, num_workers=2)
    elif args.data_set.lower() == 'oasis':

        transform_train =  torchvision.transforms.Compose([
                                                            torchvision.transforms.ToPILImage(),
                                                            torchvision.transforms.Resize(args.image_size),
                                                            torchvision.transforms.RandomCrop(args.image_size, padding=16),
                                                            torchvision.transforms.RandomAffine((-15,15), shear=(-5,5)),
                                                            torchvision.transforms.RandomHorizontalFlip(),
                                                            torchvision.transforms.ToTensor(),
                                                        ])

        transform_val = torchvision.transforms.Compose([
                                                        torchvision.transforms.ToPILImage(),
                                                        torchvision.transforms.Resize(args.image_size),
                                                        torchvision.transforms.ToTensor(),
                                                    ])

        train_set    = build_oasis(root='/scratch/backUps/jzopes/data/oasis_project/Transformer/', train=True, transform=None)#transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_set      = build_oasis(root='/scratch/backUps/jzopes/data/oasis_project/Transformer/', train=False, transform=None)#transform_val)
        val_loader   = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif 'mprage' in args.data_set.lower():
        transform_train =  transforms.Compose([
                                        Resize(args.image_size),
                                        RandomZoom(.15),
                                        RandomHorizontalFlip(.5)
                                    ]) if args.aug else \
                           Resize(args.image_size)

        transform_val = Resize(args.image_size)
        
        train_set    = build_mprage(root='/scratch/mplatscher/imaging_data/', train=True, train_size=0.8, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_set      = build_mprage(root='/scratch/mplatscher/imaging_data/', train=False, train_size=0.8, transform=transform_val)
        val_loader   = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError("Data set {} unknkown!".format(args.data_set))

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


    model = VT(args).to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
    
    train(model, train_loader, val_loader, device, criterion, optimizer, scheduler, args.epochs, args.classes, get_weights(train_set), args.patience, args.name)
    
    torch.save(model.state_dict(), 'data/' + args.name + '.pt')
    

if __name__ == '__main__':

    args = parse_args()

    main(args)
