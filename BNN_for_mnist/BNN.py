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

        self.conv2_mu = nn.Parameter(torch.randn(hidden_1, 1, 7, 7))
        self.conv2_sigma = nn.Parameter(torch.randn(hidden_1, 1, 7, 7))
        self.conv2_2mu = nn.Parameter(torch.randn(32,16,7,7))
        self.conv2_2sigma = nn.Parameter(torch.randn(32,16,7,7))
        self.fc_mu = nn.Parameter(torch.randn(64,8192))
        self.fc_sigma = nn.Parameter(torch.randn(64,8192))
        
    def forward(self, x):
        conv2_w_prior = nn.Parameter(torch.normal(mean=torch.zeros_like(self.conv2.weight), std=torch.ones_like(self.conv2.weight)))
        self.conv2.weight = nn.Parameter(conv2_w_prior*self.conv2_sigma + self.conv2_mu)
        conv2_2w_prior = nn.Parameter(torch.normal(mean=torch.zeros_like(self.conv2_2.weight), std=torch.ones_like(self.conv2_2.weight)))
        self.conv2_2.weight = nn.Parameter(conv2_2w_prior*self.conv2_2sigma + self.conv2_2mu)
        fc_prior = nn.Parameter(torch.normal(mean=torch.zeros_like(self.fc1.weight), std=torch.ones_like(self.fc1.weight)))
        self.fc1.weight = nn.Parameter(fc_prior * self.fc_sigma + self.fc_mu)

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

# KL_div between 2 gaussian of multiple dims
def KL_2gaussian(mu1, sig1, mu2, sig2):
    output = torch.log(torch.abs(sig2)/torch.abs(sig1)) + (torch.square(sig1)+torch.square(mu1-mu2))/torch.square(sig2) - 0.5
    output = torch.mean(output)
    return output


net = NN(32, 16, 64, 10)
optimizer = optim.Adadelta(net.parameters(), lr=0.0005)
log_softmax = nn.LogSoftmax(dim=1)
kernel_mu = torch.zeros_like(net.conv2.weight)
kernel_sigma = torch.ones_like(net.conv2.weight)
kernel_2mu = torch.zeros_like(net.conv2_2.weight)
kernel_2sigma = torch.ones_like(net.conv2_2.weight)
kernel_fc_mu = torch.zeros_like(net.fc1.weight)
kernel_fc_sigma = torch.ones_like(net.fc1.weight)

kl = KL_2gaussian(net.conv2_mu, net.conv2_sigma, kernel_mu, kernel_sigma)

def train(model, train_data, optimizer, lumbda):
    model.train()
    for batch_id, data in enumerate(train_data):
        optimizer.zero_grad()
        output = model(data[0])
        # Now the KL_div.shape would be (16,1,7,7), for each parameter has a loss, now I compute the mean
        loss = F.nll_loss(output, data[1])  + lumbda*KL_2gaussian(net.conv2_mu, net.conv2_sigma, kernel_mu, kernel_sigma) + lumbda*KL_2gaussian(net.conv2_2mu, net.conv2_2sigma, kernel_2mu, kernel_2sigma)+lumbda*KL_2gaussian(net.fc_mu, net.fc_sigma, kernel_fc_mu, kernel_fc_sigma)
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
for epoch in range(1, 16):
    train(net, train_loader, optimizer, lumbda=0.9)
    test(net, test_loader)
    scheduler.step()

pass