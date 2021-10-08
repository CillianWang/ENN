import numpy as np
import matplotlib.pyplot as plt
import os
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_welch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

# ALICE, BOB = 0, 1

folder = "SleepEDF\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette"
# filelist = os.listdir(folder)
# length = len(filelist)

# raw_train = mne.io.read_raw_edf(folder+"\\"+filelist[0], stim_channel='marker',
#                                 misc=['rectal'])
# annot_train = mne.read_annotations(folder+"\\"+filelist[1])

# raw_train.set_annotations(annot_train, emit_warning=False)

# # plot some data
# # scalings were chosen manually to allow for simultaneous visualization of
# # different channel types in this specific dataset
# raw_train.plot(start=60, duration=60,
#                scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7,
#                              misc=1e-1))

ALICE, BOB = 0, 1

[alice_files, bob_files] = fetch_data(subjects=[ALICE, BOB], recording=[1], update_path=True)

raw_train = mne.io.read_raw_edf(alice_files[0], stim_channel='marker',
                                misc=['rectal'])
annot_train = mne.read_annotations(alice_files[1])

raw_train.set_annotations(annot_train, emit_warning=False)

# plot some data
# scalings were chosen manually to allow for simultaneous visualization of
# different channel types in this specific dataset
raw_train.plot(start=60, duration=60,
               scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7,
                             misc=1e-1))
pass