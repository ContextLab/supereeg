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
