# -*- coding: utf-8 -*-
from . import helpers

def predict(bo, model=None):
    """
    Takes a brain object and a 'full' covariance model and fills in all
    electrode locations

    Parameters
    ----------

    bo : Brain data object
        The brain data object that you want to predict

    model : Pandas DataFrame
        An electrode x electrode 'full' covariance matrix containing all desired
        electrode locations

    Returns
    ----------

    bo_p : Brain data object
        New brain data object with missing electrode locations filled in

    """
