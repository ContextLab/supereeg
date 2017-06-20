import os
import sys
import pickle
import numpy as np
from .brain import Brain
from .model import Model
from ._helpers.stats import tal2mni
from ._helpers.stats import z2r
from scipy.spatial.distance import squareform

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

        with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/average_model_k_10_r_20.npz', 'rb') as handle:
            f = np.load(handle)
            model = squareform(f['matrix_sum'].flatten(), checks=False)
            model[np.eye(model.shape[0]) == 1] = 0
            model[np.where(np.isnan(model))] = 0
            # model = z2r(model)
            n_subs = squareform(f['weights_sum'], checks=False)

        with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/R_small_MNI.npy', 'rb') as handle:
            locs = np.load(handle)

        return Model(data=model, locs=locs, n_subs=n_subs)
