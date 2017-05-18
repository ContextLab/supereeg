# -*- coding: utf-8 -*-
from ._helpers.stats import *
from .brain import Brain

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
    sub_corrmat = get_corrmat(bo)

    # get rbf weights
    sub_rbf_weights = rbf(pd.concat([model.locs,bo.locs]), bo.locs)

    #  get subject expanded correlation matrix
    sub_corrmat_x = get_expanded_corrmat(sub_corrmat, sub_rbf_weights)

    # expanded rbf weights
    model_rbf_weights = rbf(pd.concat([model.locs,bo.locs]), model.locs)

    # get model expanded corrlation matrix
    model_corrmat_x = get_expanded_corrmat(model.data.as_matrix(), model_rbf_weights)

    # add in new subj data
    model_corrmat_x = ((model_corrmat_x*model.n_subs)+sub_corrmat_x)/model.n_subs+1

    # timeseries reconstruction
    reconstructed = reconstruct_activity(bo, model_corrmat_x)

    # # create new bo with inferred activity
    reconstructed_bo = Brain(data=reconstructed, locs=pd.concat([model.locs,bo.locs]),
                sessions=bo.sessions, sample_rate=bo.sample_rate)

    return reconstructed_bo
