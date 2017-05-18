# -*- coding: utf-8 -*-
#from . import helpers
import helpers
from _helpers.stats import *

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

    # get subject-specific correlation matrix
    corrmat = get_corrmat(bo)

    # get rbf weights
    weights = rbf(model.locs, bo.locs)

    #  get full correlation matrix
    corrmat_x = get_full_corrmat(corrmat, weights)

    # timeseries reconstruction
    # cxi = infer_activity(cx)

    # # create new bo with inferred activity
    # boi = Brain(data=cxi, locs=None)

    return corrmat_x
