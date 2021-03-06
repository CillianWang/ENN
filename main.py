import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from DataLoader import HaaglandenDataset, getDataLoaders
import torch.optim as optim
# from RecordingLoader import HaaglandenDataset_truth, getDataLoaders_truth
from Library.Models.CNN import Passthrough
from Library.cb_loss import CB_loss, count_class


# cuda device:

Train_Data = HaaglandenDataset("Train")
# Train_Truth = HaaglandenDataset_truth("Train")

batch_size =32
dataloader = DataLoader(Train_Data, batch_size=batch_size,shuffle=False,drop_last=False)
# recordingloader = DataLoader(Train_Truth, batch_size=32,shuffle=False, drop_last=False)

model = Passthrough().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(dataloader, batch_size, epochs):
    for epoch in range(epochs):
        correct = 0
        count = 0
        total_loss = 0
        index = 0
        for dl, rl in dataloader:
            batch_correct = 0
            dl, rl = dl.cuda(), rl.cuda()
            if rl.max()==5:
                continue
            optimizer.zero_grad()
            output = model(dl)

            if output.shape[0] != batch_size:
                continue
            
            samples_per_cls = count_class(rl, 5)
            loss = CB_loss(rl.reshape(batch_size,).cpu(), output.cpu(), samples_per_cls, 5, "focal", 0.9999, 2)
            #loss = F.nll_loss(output.reshape(batch_size, 5, 1), rl)

            # rl = F.one_hot(rl.to(torch.int64), 5)
            loss.backward()
            optimizer.step()
            # print(str(epoch)+":"+"loss equals to"+str(float(loss)))

            total_loss += loss


            estimate = output.argmax(-1)
            count += batch_size
            for i in range(batch_size):
                if int(estimate[i])==int(rl[i]):
                    correct += 1
                    batch_correct +=1
            
            if index%150 == 0:
                print("20 batch's acc:"+str(batch_correct))
            index += 1
        
        acc = correct/count
        print("loss:" + str(total_loss))
        print(str(acc))

            

train(dataloader, batch_size, 5)




pass