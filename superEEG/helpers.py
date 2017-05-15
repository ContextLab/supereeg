import numpy as np

# CHANGE THIS TO TAKE A BO
def apply_by_file_index(fname, xform, aggregator, field='Y'):

    """
    Session dependent function application and aggregation

    Parameters
    ----------
    fname : Data matrix (npz file)
        The data to be analyzed.
        Filename containing fields:
            Y - time series
            R - electrode locations
            fname_labels - session number
            sample_rate - sampling rate

    xform : function
        The function to apply to the data matrix from each filename

    aggregator: function
        Tunction for aggregating results across multiple iterations

    Returns
    ----------
    results : numpy ndarray
         Array of aggregated results

    """

    data = np.load(fname, mmap_mode='r')
    file_inds = np.unique(data['fname_labels'])

    results = []
    for i in file_inds:
        if np.shape(data['fname_labels'])[1] == 1:
            fname_labels = data['fname_labels'].T
        else:
            fname_labels = data['fname_labels']
        next_inds = np.where(fname_labels == i)[1]
        next_vals = xform(data[field][next_inds, :])
        if len(results) == 0:
            results = next_vals
        else:
            results = aggregator(results, next_vals)
    return results

def aggregate(prev, next):
    return np.sum(np.concatenate((prev[:, :, np.newaxis], next[:, :, np.newaxis]), axis=2), axis=2)

def zcorr(x):
    return r2z(1 - squareform(pdist(x.T, 'correlation')))

import requests
import pickle
import pandas as pd
import sys

def load_example_data():
    """
    Load example data

    Returns
    ----------
    data : Numpy Array
        Example data

    """
    if sys.version_info[0]==3:
        pickle_options = {
            'encoding' : 'latin1'
        }
    else:
        pickle_options = {}

    if dataset is 'weights':
        fileid = '0B7Ycm4aSYdPPREJrZ2stdHBFdjg'
        url = 'https://docs.google.com/uc?export=download&id=' + fileid
        data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
    if dataset is 'weights_avg':
        fileid = '0B7Ycm4aSYdPPRmtPRnBJc3pieDg'
        url = 'https://docs.google.com/uc?export=download&id=' + fileid
        data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
    if dataset is 'weights_sample':
        fileid = '0B7Ycm4aSYdPPTl9IUUVlamJ2VjQ'
        url = 'https://docs.google.com/uc?export=download&id=' + fileid
        data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
    elif dataset is 'spiral':
        fileid = '0B7Ycm4aSYdPPQS0xN3FmQ1FZSzg'
        url = 'https://docs.google.com/uc?export=download&id=' + fileid
        data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)
    elif dataset is 'mushrooms':
        fileid = '0B7Ycm4aSYdPPY3J0U2tRNFB4T3c'
        url = 'https://docs.google.com/uc?export=download&id=' + fileid
        data = pd.read_csv(url)

    return data
