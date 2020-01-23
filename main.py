# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from configs.config import config
import networks.cifar as models
import torch.optim as optim
import torch.nn as nn
from utils.utils import build_logging
from utils.utils import save_checkpoint
from utils.functions import train
from utils.functions import test
from utils.functions import evaluate
from utils.functions import save_data
from utils.dataset_utils import resized_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',
                        default='vgg16_bn',
                        type=str,
                        help='model architecture')
    parser.add_argument('--dataset',
                        default='cifar10',
                        type=str,
                        help='name of dataset')
    parser.add_argument('--pretrain',
                        default=0,
                        type=int,
                        help='Number of pretraining epochs using the cross entropy loss, so that the learning can always start. Note that it defaults to 100 if dataset==cifar10 and reward<6.1, and the results in the paper are reproduced.')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    for i in range(len(config.rewards_list)):
        reward = config.rewards_list[i]
        config.output_dir = os.path.join('output', 'o_{:.2f}'.format(reward))
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        config.model_dir = os.path.join(config.output_dir, 'model')
        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)
        config.log_dir = os.path.join(config.output_dir, 'log')
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        if args.pretrain == 0 and reward < 6.1:
            if args.dataset == 'cifar10':
                args.pretrain = 100
            elif args.dataset == 'svhn':
                args.pretrain = 50
        config.arch = args.arch
        config.dataset = args.dataset
        config.pretrain = args.pretrain
        logger = build_logging(config)

        if args.dataset == 'cifar10':
            num_classes = 10
            input_size = 32
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
            train_dataset = resized_dataset(train_dataset, train_transform, resize=32)
            test_dataset = resized_dataset(test_dataset, test_transform, resize=32)
        elif args.dataset == 'svhn':
            num_classes = 10
            input_size = 32
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.SVHN(root='./data/svhn', split='train', download=True)
            test_dataset = datasets.SVHN(root='./data/svhn', split='test', download=True)
            train_dataset = resized_dataset(train_dataset, train_transform, resize=32)
            test_dataset = resized_dataset(test_dataset, test_transform, resize=32)
        elif args.dataset == 'catsdogs':
            num_classes = 2
            input_size = 64
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=6),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            train_dataset = datasets.ImageFolder('./data/cats_dogs/train')
            test_dataset = datasets.ImageFolder('./data/cats_dogs/test')
            # resizing the images to 64 and center crop them, so that they become 64x64 squares
            train_dataset = resized_dataset(train_dataset, transform_train, resize=64)
            test_dataset = resized_dataset(test_dataset, transform_test, resize=64)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.num_workers)

        title = args.dataset + '-' + args.arch + ' o={:.2f}'.format(reward)
        model = models.__dict__[args.arch](
            num_classes=num_classes+1, input_size=input_size).to(config.device)
        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum, weight_decay=config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.steps, gamma=config.gamma)
        if args.pretrain:
            criterion = nn.CrossEntropyLoss().to(config.device)
        else:
            criterion = None
        weight = np.ones(train_dataset.__len__())
        best_metric = 0.
        for epoch in range(config.epochs):
            train_epoch_loss, train_epoch_metric, weight = train(config, epoch, model, train_loader, criterion, optimizer, logger, reward, weight)
            test_epoch_loss, test_epoch_metric = test(config, epoch, model, test_loader, criterion, logger, reward)
            if np.mean(test_epoch_metric['top1']) > best_metric:
                best_metric = np.mean(test_epoch_metric['top1'])
                evaluate(config, model, test_loader)
                save_data(config)
            logger.info('epoch complete')
            lr_scheduler.step()

        evaluate(config, model, test_loader)
        save_data(config)

        save_checkpoint({
            'epoch': config.epochs + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, config.model_dir)

if __name__ == '__main__':
    main()

