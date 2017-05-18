import numpy as np
import os
from .brain import Brain
from .model import Model
from ._helpers.stats import tal2mni

def load_example_data():

    with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/BW001.npz', 'rb') as handle:
        f = np.load(handle)
        data = f['Y']
        sample_rate = f['samplerate']
        sessions = f['fname_labels']
        locs = tal2mni(f['R'])

    return Brain(data=data, locs=locs, sessions=sessions, sample_rate=sample_rate)

def load_example_model():

    with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/R_small_MNI.npy', 'rb') as handle:
        locs = np.load(handle)

    return Model(data=None, locs=locs)
