import pandas as pd
import seaborn as sns

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

    Methods
    ----------

    plot : function
        Plots model as a heatmap

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

        self.plot = self.plot

    def plot(self):
        """
        Plot the superEEG model
        """
        sns.heatmap(self.data, xticklabels=False, yticklabels=False)
        sns.plt.title('SuperEEG Model, N=' + str(self.n_subs))
        sns.plt.show()
