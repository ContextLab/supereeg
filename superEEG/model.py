import pandas as pd
import seaborn as sns

class Model(object):
    """
    Class containing superEEG model and associated locations


    Parameters
    ----------

    data : 2d numpy array
        electrodes x electrodes correlation matrix

    locs : 1d numpy array or list
        MNI coordinate (x,y,z) by number of electrode df containing electrode locations

    meta : dict
        Optional dict containing whatever you want
    """

    def __init__(self, data=None, locs=None, meta={}):

        # convert data to df
        self.data = pd.DataFrame(data)

        # locs
        self.locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

        # meta
        self.meta = meta

        self.plot = self.plot

    def plot(self):
        sns.heatmap(self.data)
        sns.plt.show()
