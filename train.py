# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.empty_cache()
time_list = []

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        labels = labels.to(device)
        images = images.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

    finish = time.time()
    time_list.append(finish - start)
    
@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    # parser.add_argument('-world_size', type=int, default=1, help='gpus')
    args = parser.parse_args()
    args = parser.parse_args()
    net = get_network(args)
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    # print('world_size: ', world_size)
    # print('local_rank: ', local_rank)
    # torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    if local_rank == 0:
        print('parameters count: ',count_parameters(net))
    net = net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=0, find_unused_parameters=False)
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
    ])
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_sampler = torch.utils.data.distributed.DistributedSampler(cifar100_training)
    cifar100_training_loader = DataLoader(
    cifar100_training, shuffle=False, num_workers=4, batch_size=args.b,sampler=cifar100_training_sampler)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    best_acc = 0.0
    for epoch in range(1, settings.EPOCH + 1):
        cifar100_training_sampler.set_epoch(epoch)
        train(epoch)
        if local_rank == 0:
            acc = eval_training(epoch)
        train_scheduler.step(epoch)
    if local_rank ==0:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
        print(np.round(np.array(time_list).sum() / len(time_list),3))

