"""
A basic CNN model
/**
 * @author Xinping Wang
 * @email [x.wang3@student.tue.nl]
 * @create date 2021-09-11 09:32:41
 * @modify date 2021-09-11 09:32:41
 * @desc [description]
 */
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


from imageio import imread


classes = ("W", "R", "N1", "N2", "N3")

class Passthrough(nn.Module):
    def __init__(self, n_channels=8, n_classes=5):
        super().__init__()
        
        self.conv1 = nn.Conv1d(8, 16, 15)
        self.conv2 = nn.Conv1d(16, 32, 9)
        self.conv3 = nn.Conv1d(32, 64, 5)
        self.conv4 = nn.Conv1d(64, 128, 7, stride=3)
        self.conv5 = nn.Conv1d(128, 64, 7, stride=3)
        self.flat = nn.Flatten(1, -1)
        self.fc = nn.Linear(3840, 5)
        self.maxpool = nn.MaxPool1d(5, stride=5)
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, x):
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv1(x))))
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv2(x))))
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv3(x))))
        x = F.softmax(self.fc(self.flat(x)))

        # x = nn.MaxPool1d(5,5)
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.softmax(self.fc(self.flat(x)), dim=1)

        return x


