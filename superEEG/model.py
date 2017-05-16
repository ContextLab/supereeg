import pandas as pd

class Model(object):
    """
    Class containing superEEG model and associated locations
    """
    def __init__(self, data=None, locs=None):

        # convert data to df
        self.data = pd.DataFrame(data)

        # locs
        self.locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])
