import os
import sys
import pickle
import numpy as np
from .brain import Brain
from .model import Model
from ._helpers.stats import tal2mni

def load(dataset):
    """
    Load example data

    Parameters
    ----------
    dataset : string
        The name of the example data or a filepath

    Returns
    ----------
    data : any
        Example data

    """
    if sys.version_info[0]==3:
        pickle_options = {
            'encoding' : 'latin1'
        }
    else:
        pickle_options = {}

    # if dataset is 'example_data':
    #     fileid = '0B7Ycm4aSYdPPREJrZ2stdHBFdjg'
    #     url = 'https://docs.google.com/uc?export=download&id=' + fileid
    #     data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)

    if dataset is 'example_data':
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/BW001.npz', 'rb') as handle:
            f = np.load(handle)
            data = f['Y']
            sample_rate = f['samplerate']
            sessions = f['fname_labels']
            locs = tal2mni(f['R'])

        return Brain(data=data, locs=locs, sessions=sessions, sample_rate=sample_rate)

    elif dataset is 'example_model':
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/R_small_MNI.npy', 'rb') as handle:
            locs = np.load(handle)
            data = np.random.rand(len(locs), len(locs))

        return Model(data=data, locs=locs, n_subs=67)
