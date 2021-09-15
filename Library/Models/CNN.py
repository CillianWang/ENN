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
from torch.optim.lr_scheduler import StepLR

classes = ("W", "R", "N1", "N2", "N3")

class Passthrough(nn.Module):
    def __init__(self, n_channels=8, n_classes=5):
        super().__init__()
        
        self.conv1 = nn.Conv1d(8, 16, 32, stride=3)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=3)
        # self.conv3 = nn.Conv1d(32, 64, 32, stride=3)
        self.flat = nn.Flatten(1, -1)
        self.fc = nn.Linear(26880, 5)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = F.softmax(self.fc(self.flat(x)), dim=1)

        return x


