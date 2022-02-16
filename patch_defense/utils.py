# Adversarial Patch: utils
# Utils in need to generate the patch and test on the dataset.
# Created by Junbo Zhao 2020/3/19

import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
import sys
import os

# Load the datasets
# We randomly sample some images from the dataset, because ImageNet itself is too large.
# def dataloader(train_size, test_size, data_dir, batch_size, num_workers, total_num=50000):
def dataloader(train_size, test_size, batch_size, num_workers, total_num=50000):
    # Setup the transformation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    index = np.arange(total_num)
    np.random.shuffle(index)
    train_index = index[:train_size]
    test_index = index[train_size: (train_size + test_size)]

    train_dataset = torchvision.datasets.ImageFolder(root='./datasets/imagenet', transform=train_transforms)
    # print("train_dataset", train_dataset)
    test_dataset = torchvision.datasets.ImageFolder(root='./datasets/imagenet', transform=test_transforms)
    # print("test_dataset", test_dataset.class_to_idx)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_index), num_workers=num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_index), num_workers=num_workers, pin_memory=True, shuffle=False)
    # print("dataloader", train_loader)
    return train_loader, test_loader

# # Test the model on clean dataset
# def test(model, dataloader):
#     model.eval()
#     correct, total, loss = 0, 0, 0
#     with torch.no_grad():
#         for (images, labels) in dataloader:
#         # for i, data in enumerate(dataloader):
#             # images, labels = data
#             images = images.cuda()
#             labels = labels.cuda()
#             outputs = model(images)

#             _, predicted = torch.max(outputs.data, 1)
#             # print("predict is", predicted)
#             total += labels.shape[0]
#             correct += (predicted == labels).sum().item()
#     return correct / total

# # Load the log and generate the training line
# def log_generation(log_dir):
#     # Load the statistics in the log
#     epochs, train_rate, test_rate = [], [], []
#     with open(log_dir, 'r') as f:
#         reader = csv.reader(f)
#         flag = 0
#         for i in reader:
#             if flag == 0:
#                 flag += 1
#                 continue
#             else:
#                 epochs.append(int(i[0]))
#                 train_rate.append(float(i[1]))
#                 test_rate.append(float(i[2]))

#     # Generate the success line
#     plt.figure(num=0)
#     plt.plot(epochs, test_rate, label='test_success_rate', linewidth=2, color='r')
#     plt.plot(epochs, train_rate, label='train_success_rate', linewidth=2, color='b')
#     plt.xlabel("epoch")
#     plt.ylabel("success rate")
#     plt.xlim(-1, max(epochs) + 1)
#     plt.ylim(0, 1.0)
#     plt.title("patch attack success rate")
#     plt.legend()
#     plt.savefig("training_pictures/patch_attack_success_rate.png")
#     plt.close(0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f