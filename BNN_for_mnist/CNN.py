"""
####################################
same struture CNN and BNN comparison
/**
 * @author Xinping Wang
 * @email [x.wang3@student.tue.nl]
 * @create date 2021-08-22 11:34:16
 * @modify date 2021-08-22 11:34:16
 * @desc [description]
 */
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import display
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from imageio import imread
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.optim.lr_scheduler import StepLR
# from mnist_loader import mnist_data

device = torch.device("cuda")
# import mnist data
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist-data/', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),])),
        batch_size=32, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist-data/', train=False, transform=transforms.Compose([transforms.ToTensor(),])
                       ),
        batch_size=32, shuffle=True)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

class NN(nn.Module):
    
    def __init__(self, batch_size, hidden_1, hidden_2, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(32*16*16, hidden_2)
        self.hidden_1 = hidden_1
        self.batch_norm_1 = nn.BatchNorm2d(hidden_1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=hidden_1, kernel_size=7,stride=1)
        self.conv2_2 = nn.Conv2d(in_channels=hidden_1, out_channels=32, kernel_size = 7)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.out = nn.Linear(32*16*16, output_size)
        
    def forward(self, x):
        output = self.batch_norm_1(self.conv2(x))
        output = F.relu(output)
        output = self.batch_norm_2(self.conv2_2(output))
        output = F.relu(output)
        output = output.flatten(1)
        # x = x.view(-1, 28*28)
        # output = self.fc1(output)
        # output = F.relu(output)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output

net = NN(32, 16, 64, 10)
optimizer = optim.Adadelta(net.parameters(), lr=0.001)
log_softmax = nn.LogSoftmax(dim=1)

def train(model, train_data, optimizer, epoch):
    model.train()
    for batch_id, data in enumerate(train_data):
        optimizer.zero_grad()
        output = model(data[0])
        loss = F.nll_loss(output, data[1])
        loss.backward()
        optimizer.step()

def test(model, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_id, data in enumerate(test_data):
            output = model(data[0])
            test_loss += F.nll_loss(output, data[1], reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(data[1].view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


scheduler = StepLR(optimizer, step_size=1)
for epoch in range(1, 6):
    train(net, train_loader, optimizer, epoch)
    test(net, test_loader)
    scheduler.step()