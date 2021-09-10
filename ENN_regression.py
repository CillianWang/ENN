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
import matplotlib.pyplot as plt
import torch.nn as nn
# Setting cuda
device = torch.device("cuda" if torch.cuda.is_available else "cpu")


x = np.linspace(-10,10, 1000) # step would be 0.021

noise = np.random.normal(0, 1, 1000)
noise_index = np.where(x<3, 1, 7)
noise = noise_index*noise

class ENN(nn.Module):

    def __init__(self, z_mu, z_sigma):
        super(ENN, self).__init__()
        self.a_mu = nn.Parameter(torch.randn(1))
        self.a_mu.requires_grad = True
        self.a_sigma = nn.Parameter(torch.randn(1))
        self.a_sigma.requires_grad = True
        self.b_mu = nn.Parameter(torch.randn(1))
        self.b_mu.requires_grad = True
        self.b_sigma = nn.Parameter(torch.randn(1))
        self.b_sigma.requires_grad = True

        # self.M = M
        self.z_mu = z_mu
        self.z_sigma = z_sigma

        # Linear layer regression:
        self.depth = 64
        self.linear1 = nn.Linear(1, self.depth, bias=True)
        self.linear2 = nn.Linear(self.depth, 1, bias=True)

        self.mu1 = nn.Parameter(torch.randn(self.depth, 1))
        self.mu1.requires_grad = True
        self.mu2 = nn.Parameter(torch.randn(1, self.depth))
        self.mu2.requires_grad = True
        self.sigma1 = nn.Parameter(torch.randn(self.depth, 1))
        self.sigma1.requires_grad = True
        self.sigma2 = nn.Parameter(torch.randn(1, self.depth))
        self.sigma2.requires_grad = True




    def forward(self, x):
        output = 0
        linear1_prior = torch.normal(mean=torch.zeros_like(self.linear1.weight)+self.z_mu, std=torch.ones_like(self.linear1.weight)*self.z_sigma)
        # self.linear1.weight = nn.Parameter((linear1_prior * self.sigma1 + self.linear1.weight))

        linear2_prior = torch.normal(mean=torch.zeros_like(self.linear2.weight)+self.z_mu, std=torch.ones_like(self.linear2.weight)*self.z_sigma)
        # self.linear2.weight = nn.Parameter((linear2_prior * self.sigma2 + self.linear2.weight))


        # for i in range(self.M):
        # # simple y = a*x regression:
        #     # output += x * (torch.normal(mean=self.z_mu,std=torch.Tensor([self.z_sigma]))*self.a_sigma +self.a_mu)
        # # linear layer regression:
        #     x = torch.from_numpy(np.array(x).reshape(1,1))
        #     output += self.linear2(self.linear1(x.float()))

        #return output/self.M
        x = torch.from_numpy(np.array(x).reshape(1,1))
        output = ((linear1_prior * self.sigma1)*x).reshape(1, self.depth) + self.linear1(x.float())
        output = self.linear2(output.float()) + torch.mm(output.float(),(linear2_prior * self.sigma2).reshape(self.depth,1))
        output = output + torch.normal(mean=self.z_mu,std=torch.Tensor([self.z_sigma]))*self.b_sigma +self.b_mu
        return output


a = 1
net = ENN(0, 0.5).float()
b = net(a)



# A direct way to compute KL_div between two gaussian distributions:
# KL(p,q) = log(sigma2/sigma1) + (sigma1)^2+(mu1-mu2)^2/(sigma2)^2 - 1/2
def KL_2gaussian(mu1, sig1, mu2, sig2):
    output = torch.log(torch.abs(sig2)/torch.abs(sig1)) + (torch.square(sig1)+torch.square(mu1-mu2))/torch.square(sig2) - 0.5
    output = torch.mean(output)
    return output



#####################################################
#  main

N = len(x)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
kernel_mu = torch.zeros_like(net.linear1.weight)
kernel_sigma = torch.ones_like(net.linear1.weight)
kernel_2mu = torch.zeros_like(net.linear2.weight)
kernel_2sigma = torch.ones_like(net.linear2.weight)


def train(model, train_data, optimizer, noise):
    model.train()
    index = 0
    for data in train_data:
        optimizer.zero_grad()
        output = model(data)
        loss = KL_2gaussian(net.mu1, net.sigma1, kernel_2mu, kernel_2sigma) +  KL_2gaussian(net.mu2, net.sigma2, kernel_2mu, kernel_2sigma)+ torch.square(output-torch.tensor(data)*5-noise[index]-100)
        # loss = torch.square(output-torch.tensor(data)*5-noise[index])
        loss.backward()
        optimizer.step()
        index += 1
        # print(loss)

    # print(net.a_mu, net.a_sigma)

for i in range(15):
    train(net, x, optimizer, noise)




# kl = KL_2gaussian(torch.abs(net.a_mu), torch.abs(net.a_sigma), torch.tensor(4.5), torch.tensor(0.5))
# print(net.a_mu, net.a_sigma)


# Now show some figure of this regression:
y = []
for i in x:
    # simple estimation:
    # alpha = torch.normal(mean = net.a_mu, std = abs(net.a_sigma))
    # y.append(float(alpha)*i)
    out = 0
    for m in range(10):
        out += net(i)
    y.append(out/10)
    pass

plt.subplot(121)
plt.plot(x,y, color='red')
plt.title("estimated")
plt.subplot(122)
plt.plot(x,5*x+noise, color="blue")
plt.title("origin")
plt.show()
pass
        