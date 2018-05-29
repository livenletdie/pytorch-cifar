'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import os.path
import argparse
import numpy as np

from models import *
import models
from utils import progress_bar


from stochasticWeightAverageTrainer import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default = "ResNet18", type = str, help = "Model to run")
parser.add_argument('--lrEpoch', default = 30, type = int, help = "Epochs where the LR is reduced by 10")
parser.add_argument('--startEpoch', default = 0, type = int, help = "Start Epoch Count")
parser.add_argument('--nEpochs', default = 100, type = int, help = "Total Epoch Count")
parser.add_argument('--cosineLR', action = 'store_true', help = 'use cosine LR')
parser.add_argument('--cosineCycleCount', default = 1, type = int, help = 'Number of cycles through cos LR')
parser.add_argument('--lrCycleScale', default = 1.0, type = float, help = 'Starting LR scale across the cos cycles')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--cifar100', action = 'store_true', help = "Train with CIFAR100. Default is train with CIFAR10")
parser.add_argument('--save', default = "checkpoint", type = str, help = "Directory to save the checkpoints")
parser.add_argument('--swaStartEpoch', default = -1, type = int, help = "Start Epoch for SWA")
parser.add_argument('--swaEndEpoch', default = -1, type = int, help = "End epoch for SWA. I recomment one epoch of training with almost zero LR to fix the BN statistics after SWA")
parser.add_argument('--lrswa', default = 0.001, type = float, help = "Learning rate for SWA.")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = args.startEpoch  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
classCount = 10

if args.cifar100:
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    classCount = 100
#end if

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model:', args.model)
# net = VGG('VGG19')
if hasattr(models, args.model) == False:
    print("ERROR: Model Not Found", dir(models))
    sys.exit(1)
#end if
modelObj = getattr(models, args.model)
net = modelObj(classCount)

# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#end if

swaManager = StochasticWeightAvgTrainer(net, args.swaStartEpoch, args.swaEndEpoch, args.lrswa)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.save), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(args.save, 'ckpt.t7'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] if start_epoch < 0 else start_epoch

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save):
            os.mkdir(args.save)
        torch.save(state, os.path.join(args.save, 'ckpt.t7'))
        best_acc = acc
    print("TEST_INFO:", epoch, acc, best_acc)

#end


def adjust_learning_rate_step(optimizer, lr, epoch, lrStep):
    lr = lr * (0.1 ** (epoch //lrStep))
    print("LEARNING_RATE:", epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #end for
#end adjust_learning_rate

def adjust_learning_rate_cos(optimizer, lr, epoch, nEpochs, cycleCount, lrCycleScale):
    nEpochs = nEpochs // cycleCount
    cycleIdx = epoch // nEpochs
    epoch = epoch % nEpochs
    lr = lr * (lrCycleScale ** cycleIdx) * (np.cos(epoch * np.pi/nEpochs) + 1)/2
    print("LEARNING_RATE:", epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #end for
#end adjust_learning_rate

def adjust_learning_rate_swa(optimizer, lr):
    print("LEARNING_RATE (SWA):", epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #end for
#end adjust_learning_rate_swa

def adjust_learning_rate(optimizer, epoch, swaManager, args):
    if swaManager.isSwaMode(epoch):
        adjust_learning_rate_swa(optimizer, args.lrswa)
    elif args.cosineLR:
        adjust_learning_rate_cos(optimizer, args.lr, epoch, args.nEpochs, args.cosineCycleCount, args.lrCycleScale)
    else:
        adjust_learning_rate_step(optimizer, args.lr, epoch, args.lrEpoch)
    #end if
#end adjust_learning_rate



for epoch in range(start_epoch, start_epoch+args.nEpochs):
    adjust_learning_rate(optimizer, epoch, swaManager, args)
    train(epoch)
    swaManager.swaStep(net, epoch)
    test(epoch)
#end
