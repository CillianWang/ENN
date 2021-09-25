# imports
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import os
import time
from natsort import natsorted


class HaaglandenDataset(Dataset):
    def __init__(self, split, transform=None):
        self.split = split
        self.data_loc = "data_cut"+"/"+split+"/"
        self.transform = transform
        
        # get a list of all file names
        _, self.index, _ = next(os.walk(self.data_loc))
        self.files = []
        self.truths = []
        for i in self.index:
            _, _, files = next(os.walk(self.data_loc+str(i)+"/data"))
            for file in files:
                file = os.path.join(str(i)+"/data/"+file)
                self.files.append(file)
            
            _, _, truths = next(os.walk(self.data_loc+str(i)+"/truth"))
            for truth in truths:
                truth = os.path.join(str(i)+"/truth/"+truth)
                self.truths.append(truth)
        
        # sort the files correctly (because leading zeros are ignored)
        #self.files = natsorted(self.files)
        #self.truths = natsorted(self.truths)
        
        self.file_count = len(self.files)
        self.truths_count = len(self.truths)
        
    def __len__(self):
        return self.file_count

    def __getitem__(self, idx):
        # find out what file is being fetched
        map = {" Sleep stage W": 0, 
        " Sleep stage R": 1,
        " Sleep stage N1": 2,
        " Sleep stage N2": 3,
        " Sleep stage N3": 4,
        "None": 5}


        file_name = self.files[idx]
        if idx == self.truths_count-1:
            truth_name = self.truths[idx]
        else:
            truth_name = self.truths[idx+1]

        
        # load the npy  file
        npy_data = np.load(self.data_loc+file_name)
        if npy_data.shape[-1] != 30*256:
            npy_data = np.zeros([8, 30*256])
        npy_truth = np.load(self.data_loc+truth_name)
        if map.get(npy_truth[-2]) == None:
            npy_truth[-2] = "None"
        npy_truth[-2] = map.get(npy_truth[-2])

        # get out the data
        data = torch.Tensor(npy_data)
        truth = torch.Tensor([int(npy_truth[-2])]).long()
        # hypnogram = torch.LongTensor([int(npy_data['hypnogram'])])
        
        return data, truth


# %% Dataloader
def getDataLoaders(args):
    # create the three data loaders
    train_set = HaaglandenDataset("Train")
    dataloader_train = DataLoader(train_set,
                                  batch_size=args.batchsize,
                                  shuffle=True, drop_last=False,
                                  pin_memory=True)
    
    dataloader_test  = DataLoader(HaaglandenDataset("Test"),
                                  batch_size=args.batchsize_test,
                                  shuffle=False, drop_last=False,
                                  pin_memory=True)
    
    dataloader_val   = DataLoader(HaaglandenDataset("Validation"),
                                  batch_size=args.batchsize_test,
                                  shuffle=False, drop_last=False,
                                  pin_memory=True)
    
    # calculate some additional stuff from the data
    args.no_examples = train_set.file_count
    args.batches_per_epoch = int(np.ceil(args.no_examples/args.batchsize))
    args.batch_modules = int(args.batches_per_epoch/args.no_samples_per_epoch)
    
    return dataloader_train, dataloader_test, dataloader_val, args


# %% testing
if __name__ == "__main__":
    # datasets
    Train_Data = HaaglandenDataset("Train")
    Test_Data = HaaglandenDataset("Test")
    Validation_Data = HaaglandenDataset("Validation")
    
    # %% data loader
    dataloader = DataLoader(Train_Data, batch_size=32,shuffle=True,drop_last=False)
    
    start = time.time()
    for i,(accelerometer,hypnogram) in enumerate(dataloader):
        print(f"For batch {i}, sizes are {accelerometer.size()} and {hypnogram.size()}")
    end = time.time()
    
    print(f"Going over the entired dataset took {end-start} seconds")