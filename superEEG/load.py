import numpy as np
import os
from .brain import Brain

def load_example_data():
    import superEEG

    with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/R_small_MNI.npy', 'rb') as handle:
        locs = np.load(handle)

    with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/BW001.npz', 'rb') as handle:
        f = np.load(handle)
        data = f['Y']
        sample_rate = f['samplerate']
        sessions = f['fname_labels']

    return Brain(data=data, locs=locs, sessions=sessions, sample_rate=sample_rate)
