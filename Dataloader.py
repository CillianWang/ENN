import mne
import os
folder = "haaglanden-medisch-centrum-sleep-staging-database-1.0.0\haaglanden-medisch-centrum-sleep-staging-database-1.0.0/recordings"
filenames = os.listdir(folder)
data_list = []
recording_list = []
index = 0
for file in filenames:
    if index%3 == 0:
        data_list.append(file)
    if index%3 == 2:
        recording_list.append(file)
    index += 1
pass
