# -*- coding: utf-8 -*-
#from . import helpers
import helpers
from _helpers.stats import *


# CHANGE THIS TO ACCEPT NUMPY ARRAY
def corrmat(bo):
    """
    Function that calculates the average subject level correlation matrix for filename across session

    Parameters
    ----------
    fname :  Data matrix (npz file)
        The data to be analyzed.
        Filename containing fields:
            Y - time series
            R - electrode locations
            fname_labels - session number
            sample_rate - sampling rate

    Returns
    ----------
    results: 2D ndarray(len(R_subj)xlen(R_subj)) matrix
        The average correlation matrix across sessions


    """
    def aggregate(prev, next):
        return np.sum(np.concatenate((prev[:, :, np.newaxis], next[:, :, np.newaxis]), axis=2), axis=2)

    def zcorr(x):
        return r2z(1 - squareform(pdist(x.T, 'correlation')))

    summed_zcorrs = apply_by_file_index(bo, zcorr, aggregate)
    n = n_files(bo)

    return z2r(summed_zcorrs / n)


def predict(bo, model=None):
    """
    Takes a brain object and a 'full' covariance model, fills in all
    electrode timeseries for all missing locations and returns the new brain object

    Parameters
    ----------

    bo : Brain data object or a list of Brain objects
        The brain data object that you want to predict

    model : Pandas DataFrame
        An electrode x electrode 'full' covariance matrix containing all desired
        electrode locations

    Returns
    ----------

    bo_p : Brain data object
        New brain data object with missing electrode locations filled in

    """

    # if model is None:
    #
    #     assert type(bo) is list, "To calculate the model, you need more than 1 brain object"

    # get subject-specific covariance matrix
    x = get_corrmat(data)

    # get full covmats
    x_e = expand_corrmat(x)

    # timeseries reconstruction
    x_r = infer_activity(xe)

    return x_r
