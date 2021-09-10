import mne
import os

folder = "haaglanden-medisch-centrum-sleep-staging-database-1.0.0\haaglanden-medisch-centrum-sleep-staging-database-1.0.0/recordings"
filenames = os.listdir(folder)
data_list = []
recording_list = []
index = 0

# read data.edf and recording.txt, as recording.edf contains strange floats
for file in filenames:
    if index%3 == 0:
        data_list.append(file)
    if index%3 == 2:
        recording_list.append(file)
    index += 1

if len(data_list) != len(recording_list):
    print("Data and recording not aligned")
else:
    print("Length of data:"+str(len(data_list))+", and length of recordings:"+str(len(recording_list)))


def read_data(filename):
    data = mne.io.read_raw_edf(filename)
    raw_data = data.get_data()
    return raw_data

def read_recordings(filename):
    f = open(filename)
    rec = [line.strip().split(',') for line in f]
    return rec

pass


# save data(cut)
a = read_data(folder+"\\"+data_list[0])
b = read_recordings(folder+"\\"+recording_list[0])
import numpy as np
np.save('a', a)
c = np.load('a.npy')



def data_cut(data, recording, index):
    path = "data_cut\\"+str(index)
    if os.path.isdir(path)==False:
        os.makedirs(path)
    n = len(recording)
    for i in range(n):
        section = data[:,i*30*256:(i+1)*30*256]
        if os.path.isdir(path+"\\data")==False:
            os.makedirs(path+"\\data")
        np.save(path+"\\data\\"+"data_"+str(i), section)
        truth = recording[i]
        if os.path.isdir(path+"\\truth")==False:
            os.makedirs(path+"\\truth")
        np.save(path+"\\truth\\"+"truth_"+str(i), truth)

for i in range(154):
    a = read_data(folder+"\\"+data_list[i])
    b = read_recordings(folder+"\\"+recording_list[i])
    data_cut(a,b,i)

pass
