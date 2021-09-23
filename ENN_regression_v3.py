#### Epistemic Neural network regression model, with linear modules


from os import X_OK
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

# Setting cuda 
# device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# creating signal and noise
x_i = np.linspace(-10,20, 1000) # step would be 0.021
a = x_i[0:500]
b = x_i[699:999]
x = np.hstack((a,b))

noise = np.random.normal(0, 1, 1000)
noise_index = np.where(x_i<7, 1, 10)
noise = noise_index*noise

noise_mask = np.random.normal(0, 1, 800)
noise_index = np.where(x<7, 1, 10)
noise_mask = noise_index*noise_mask


class ENN(nn.Module):
    # inputs as index z's mu and sigma
    def __init__(self, z_mu, z_sigma, l1, l2, l3, l4):
        super(ENN, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.z_mu = z_mu
        self.z_sigma = z_sigma
        self.w_mu1, self.w_sigma1, self.b_mu1, self.b_sigma1 = self.Linear_parameter(1, l1)
        self.w_mu2, self.w_sigma2, self.b_mu2, self.b_sigma2 = self.Linear_parameter(l1, l2)
        self.w_mu3, self.w_sigma3, self.b_mu3, self.b_sigma3 = self.Linear_parameter(l2, l3)
        self.w_mu4, self.w_sigma4, self.b_mu4, self.b_sigma4 = self.Linear_parameter(l3, 2)    
        # self.w_mu3, self.w_sigma3, self.b_mu3, self.b_sigma3 = self.Linear_parameter(l2, l3)
        # self.w_mu4, self.w_sigma4, self.b_mu4, self.b_sigma4 = self.Linear_parameter(l3, l4)
        # self.w_mu5, self.w_sigma5, self.b_mu5, self.b_sigma5 = self.Linear_parameter(l4, 1)


    
    def KL_2gaussian(self, mu1, sig1, mu2, sig2):
        output = torch.log(torch.abs(sig2)/torch.abs(sig1)) + (torch.square(sig1)+torch.square(mu1-mu2))/torch.square(sig2) - 0.5
        output = torch.mean(output)
        return output

    
    def Linear_parameter(self, insize, outsize):
        # asume input of shape [1, x], no batch
        # weights:
        w_mu = nn.Parameter(torch.randn(insize, outsize))
        w_mu.requires_grad = True
        w_sigma = nn.Parameter(torch.randn(insize, outsize))
        w_sigma.requires_grad = True
        # bias:
        b_mu = nn.Parameter(torch.randn(1, outsize))
        b_mu.requires_grad = True
        b_sigma = nn.Parameter(torch.randn(1, outsize))
        b_sigma.requires_grad = True

        return w_mu, w_sigma, b_mu, b_sigma
    
    def Linear(self, x, w_mu, w_sigma, b_mu, b_sigma):
        # keep the size
        insize = w_mu.shape[0]
        outsize = w_mu.shape[1]

        w_shape = torch.empty(insize, outsize)
        b_shape = torch.empty(1, outsize)

        w_prior = torch.normal(mean=torch.zeros_like(w_shape)+self.z_mu, std=torch.ones_like(w_shape)*self.z_sigma)
        b_prior = torch.normal(mean=torch.zeros_like(b_shape)+self.z_mu, std=torch.ones_like(b_shape)*self.z_sigma)
        
        # ENN's true weight and bias
        w = w_sigma*w_prior+w_mu
        b = b_sigma*b_prior+b_mu

        loss = self.KL_2gaussian(w_mu, w_sigma, torch.zeros_like(w_shape), torch.ones_like(w_sigma))

        output = torch.mm(x.double(), w.double()) + b.double()

        return output, loss
    

    def forward(self, x):
        x = torch.from_numpy(np.array(x).reshape(1,1))

        kl_loss = 0
        # Calling modules to do linear layers:
        output, loss = self.Linear(x, self.w_mu1, self.w_sigma1, self.b_mu1, self.b_sigma1)
        kl_loss += loss
        output = F.relu(output)

        output, loss = self.Linear(output, self.w_mu2, self.w_sigma2, self.b_mu2, self.b_sigma2)
        kl_loss += loss
        output = torch.sigmoid(output)

        output, loss = self.Linear(output, self.w_mu3, self.w_sigma3, self.b_mu3, self.b_sigma3)
        kl_loss += loss
        
        # output = F.sigmoid(output)
        output, loss = self.Linear(output, self.w_mu4, self.w_sigma4, self.b_mu4, self.b_sigma4)
        # kl_loss += loss
        # output = F.relu(output)
        # output, loss = self.Linear(output, self.w_mu5, self.w_sigma5, self.b_mu5, self.b_sigma5)
        # kl_loss += loss
        kl_loss = kl_loss/(self.l1+self.l1+self.l2+self.l2*self.l1)
        
        # kl_loss = kl_loss/(self.l1 + self.l2 + self.l3 + self.l4 +self.l1+self.l2*self.l1+self.l2*self.l3 + self.l3*self.l4)
        return output, kl_loss


a = 1
net = ENN(0, 0.001, 32, 64, 32, 16)
b, c = net(a)


def KL_2gaussian(mu1, sig1, mu2, sig2):
    output = torch.log(torch.abs(sig2)/torch.abs(sig1)) + (torch.square(sig1)+torch.square(mu1-mu2))/torch.square(sig2) - 0.5
    output = torch.mean(output)
    return output


optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

def nll(y, mu, sigma):
    loss = 0.5*torch.square((y-mu)/sigma) #try to make mu close to y, and making sigma large
    loss += torch.log(abs(sigma)*2.50662827463)
    return loss


def train(model, train_data, optimizer, noise):
    model.train()
    index = 0
    for data in train_data:
        optimizer.zero_grad()
        output, kl = model(data)
        y = torch.tensor(data)*torch.tensor(data)+noise[index]+100
        loss = nll(y, output[0][0], output[0][1]) + kl
        # loss = torch.square(output-torch.tensor(data)*5-noise[index])
        loss.backward()
        optimizer.step()
        index += 1
        if index%200 == 0:
            print(loss)
        # print(loss)

    # print(net.a_mu, net.a_sigma)

for i in range(200):
    train(net, x, optimizer, noise)

# kl = KL_2gaussian(torch.abs(net.a_mu), torch.abs(net.a_sigma), torch.tensor(4.5), torch.tensor(0.5))
# print(net.a_mu, net.a_sigma)


# Now show some figure of this regression:
y = []
y_low = []
y_up = []
n = 1
for i in np.linspace(-10, 10, 1000):
    # simple estimation:
    # alpha = torch.normal(mean = net.a_mu, std = abs(net.a_sigma))
    # y.append(float(alpha)*i)
    out = 0
    out_low = 0
    out_up = 0
    for m in range(n):
        out_t,loss = net(i)
        out_low += out_t[0][0] - abs(out_t[0][1])
        out_up += out_t[0][0] + abs(out_t[0][1])
        out_t = torch.normal(mean = out_t[0][0], std = abs(out_t[0][1]))
        out += out_t
    y.append(out/n)
    y_low.append(out_low/n)
    y_up.append(out_up/n)
    pass

plt.subplot(121)
plt.scatter(np.linspace(-10, 20, 1000),np.array(y), color="red", s = 1, label="y")
plt.scatter(np.linspace(-10, 20, 1000),np.array(y_low), color="green", s = 1, label="lower bound")
plt.scatter(np.linspace(-10, 20, 1000),np.array(y_up), color="yellow", s = 1, label = "upper bound")
plt.legend()
plt.title("estimated")
plt.subplot(122)
plt.scatter(x,x*x+noise_mask+100, color="blue")
plt.title("data input")
plt.show()
pass





