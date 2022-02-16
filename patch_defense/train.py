import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import torchvision
import torchvision.transforms as transforms
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# from TinyImageNet import TinyImageNet

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse

from models.small_net import VGG
# from utils import progress_bar
from utils import*


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--num_workers', type=int, default=8, help="num_workers")
parser.add_argument('--train_size', type=int, default=10000, help="number of training images")
parser.add_argument('--test_size', type=int, default=5000, help="number of test images")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', type=int, default=20, help="total epoch")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')



trainloader, testloader = dataloader(args.train_size, args.test_size, args.batch_size, args.num_workers, 50000)



# Model
print('==> Building model..')
# net = VGG('VGG19')
net = VGG()

print(net)

net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/vgg3_tiny224.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 0.00001, last_epoch = -1)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    for param_group in optimizer.param_groups:
        print('learning rate: %f' % param_group['lr'])
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # writer.add_scalar('loss value',loss, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
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
            imgs = inputs.data.cpu().numpy()
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
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/vgg3_tiny224.pth.pth')
        best_acc = acc





for epoch in range(start_epoch, start_epoch+args.epochs):
    # a = time.time()
    train(epoch)
    scheduler.step()

    

    # print("training time is:", time.time()-a)
    test(epoch)