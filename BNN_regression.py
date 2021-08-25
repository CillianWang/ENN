"""
/**
 * @author Xinping Wang
 * @email [x.wang3@student.tue.nl]
 * @create date 2021-08-23 14:59:49
 * @modify date 2021-08-23 14:59:49
 * @desc [simple implementation of Beyesian Neural Network]
 */
"""


import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# Setting cuda
device = torch.device("cuda" if torch.cuda.is_available else "cpu")


x = np.linspace(-100,100, 10000) # step would be 0.021

noise = np.random.normal(0, 1, 10000)

class BNN(nn.Module):

    def __init__(self, M):
        super(BNN, self).__init__()
        self.a_mu = nn.Parameter(torch.randn(1))
        self.a_mu.requires_grad = True
        self.a_sigma = nn.Parameter(torch.randn(1))
        self.a_sigma.requires_grad = True
        self.M = M
    def forward(self, x):
        output = 0
        for i in range(self.M):
            output += x * torch.normal(mean=self.a_mu, std=abs(self.a_sigma))
        return output/self.M


a = 1
net = BNN(10)
b = net(a)



# A direct way to compute KL_div between two gaussian distributions:
# KL(p,q) = log(sigma2/sigma1) + (sigma1)^2+(mu1-mu2)^2/(sigma2)^2 - 1/2
def KL_2gaussian(mu1, sig1, mu2, sig2):
    output = torch.log(torch.abs(sig2)/torch.abs(sig1)) + (torch.square(sig1)+torch.square(mu1-mu2))/torch.square(sig2) - 0.5
    return output



#####################################################
#  main

N = len(x)
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

def train(model, train_data, optimizer):
    model.train()
    for data in train_data:
        optimizer.zero_grad()
        output = model(data)
        loss = KL_2gaussian(torch.abs(net.a_mu), torch.abs(net.a_sigma), torch.tensor(0), torch.tensor(1)) + torch.square(output-torch.tensor(data)*5)
        loss.backward()
        optimizer.step()

for i in range(5):
    train(net, x, optimizer)




kl = KL_2gaussian(torch.abs(net.a_mu), torch.abs(net.a_sigma), torch.tensor(0), torch.tensor(1))
print(net.a_mu, net.a_sigma)
pass
        