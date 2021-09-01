"""
/**
 * @author Xinping Wang
 * @email [x.wang3@student.tue.nl]
 * @create date 2021-08-27 23:44:43
 * @modify date 2021-08-27 23:44:43
 * @desc [description]
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

    def __init__(self, z_mu, z_sigma, M):
        super(BNN, self).__init__()
        self.a_mu = nn.Parameter(torch.randn(1))
        self.a_mu.requires_grad = True
        self.a_sigma = nn.Parameter(torch.randn(1))
        self.a_sigma.requires_grad = True
        self.M = M
        self.z_mu = z_mu
        self.z_sigma = z_sigma
    def forward(self, x):
        output = 0
        for i in range(self.M):
            output += x * (torch.normal(mean=float(self.z_mu),std=(torch.arange(float(self.z_sigma))))*self.a_sigma +self.a_mu)
        return output/self.M


a = 1
net = BNN(0, 1, 10)
b = net(a)



# A direct way to compute KL_div between two gaussian distributions:
# KL(p,q) = log(sigma2/sigma1) + (sigma1)^2+(mu1-mu2)^2/(sigma2)^2 - 1/2
def KL_2gaussian(mu1, sig1, mu2, sig2):
    output = torch.log(torch.abs(sig2)/torch.abs(sig1)) + (torch.square(sig1)+torch.square(mu1-mu2))/torch.square(sig2) - 0.5
    return output



#####################################################
#  main

N = len(x)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def train(model, train_data, optimizer):
    model.train()
    for data in train_data:
        optimizer.zero_grad()
        output = model(data)
        loss = KL_2gaussian(torch.abs(net.a_mu), torch.abs(net.a_sigma), torch.tensor(4.5), torch.tensor(1)) + torch.square(output-torch.tensor(data)*5)
        loss.backward()
        optimizer.step()
    print(net.a_mu, net.a_sigma)

for i in range(15):
    train(net, x, optimizer)




kl = KL_2gaussian(torch.abs(net.a_mu), torch.abs(net.a_sigma), torch.tensor(4.5), torch.tensor(1))
print(net.a_mu, net.a_sigma)
pass
        