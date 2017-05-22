import pandas as pd
import seaborn as sns
from ._helpers.stats import *
from .brain import Brain

class Model(object):
    """
    superEEG model and associated locations

    This class holds your superEEG model.  To create an instance, you need a model
    (an electrode x electrode correlation matrix), electrode locations in MNI
    space, and the number of subjects used to create the model.  Additionally,
    you can include a meta dictionary with any other information that you want
    to save with the model.

    Parameters
    ----------

    data : numpy.ndarray
        Electrodes x electrodes correlation matrix

    locs : numpy.ndarray
        MNI coordinate (x,y,z) by number of electrode df containing electrode locations

    n_subs : int
        Number of subjects used to create model

    meta : dict
        Optional dict containing whatever you want


    Attributes
    ----------

    data : pandas.DataFrame
        Electrodes x electrodes correlation matrix

    locs : pandas.DataFrame
        MNI coordinate (x,y,z) by number of electrode df containing electrode locations

    n_subs : int
        Number of subjects used to create model

    meta : dict
        Optional dict containing whatever you want

    Returns
    ----------
    model : superEEG.Model instance
        A model that can be used to infer timeseries from unknown locations

    """

    def __init__(self, data=None, locs=None, n_subs=None, meta={}):

        # convert data to df
        self.data = pd.DataFrame(data)

        # locs
        self.locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

        # number of subjects
        self.n_subs = n_subs

        # meta
        self.meta = meta

    def predict(self, bo, threshold=10, tf=False):
        """
        Infers timeseries from unknown locations given a brain object.

        Parameters
        ----------

        bo : superEEG.Brain
            The brain data object that you want to predict

        threshold : float
            Threshold to remove bad electrodes based on kurtosis

        tf : bool
            If True, uses Tensorflow (default is False).

        Returns
        ----------

        bo_p : Brain data object
            New brain data object with missing electrode locations filled in

        """

        # filter bad electrodes
        bo = filter_elecs(bo, measure='kurtosis', threshold=threshold)

        # get subject-specific correlation matrix
        sub_corrmat = get_corrmat(bo)

        # get rbf weights
        sub_rbf_weights = rbf(pd.concat([self.locs, bo.locs]), bo.locs)

        #  get subject expanded correlation matrix
        sub_corrmat_x = get_expanded_corrmat(sub_corrmat, sub_rbf_weights)

        # expanded rbf weights
        model_rbf_weights = rbf(pd.concat([self.locs, bo.locs]), self.locs)

        # get model expanded corrlation matrix
        model_corrmat_x = get_expanded_corrmat(self.data.as_matrix(), model_rbf_weights)

        # add in new subj data
        model_corrmat_x = ((model_corrmat_x * self.n_subs) + sub_corrmat_x) / self.n_subs+1

        # timeseries reconstruction
        if tf:
            reconstructed = reconstruct_activity_tf(bo, model_corrmat_x)
        else:
            reconstructed = reconstruct_activity(bo, model_corrmat_x)

        # # create new bo with inferred activity
        bo_p = Brain(data=reconstructed, locs=self.locs,
                    sessions=bo.sessions, sample_rate=bo.sample_rate)

        return bo_p

    def plot(self):
        """
        Plot the superEEG model as a heatmap
        """
        sns.heatmap(self.data, xticklabels=False, yticklabels=False)
        sns.plt.title('SuperEEG Model, N=' + str(self.n_subs))
        sns.plt.show()
