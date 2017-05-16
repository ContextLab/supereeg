# -*- coding: utf-8 -*-

import pandas as pd
import time
import os
import numpy as np
import pickle

class Brain(object):
    """
    Brain data object for the superEEG package

    Details about the Brain object.

    Parameters
    ----------

    data : 2d numpy array or list of lists
        Samples x electrodes df containing the EEG data

    locs : 1d numpy array or list
        MNI coordinate (x,y,z) by electrode df containing electrode locations

    session : 1d numpy array or list
        Samples x 1 array containing session identifiers

    sample_rates : float or list of floats
        Sample rate of the data. If different over multiple sessions, this is a
        list

    meta : dict
        Optional dict containing whatever you want

    Attributes
    ----------

    data : Pandas DataFrame
        Samples x electrodes df containing the EEG data

    locs : Pandas DataFrame
        MNI coordinate (x,y,z) by electrode df containing electrode locations

    sessions : Pandas Series
        Samples x 1 array containing session identifiers.  If a singleton is passed,
         a single session will be created.

    sample_rates : float or list of floats
        Sample rate of the data. If different over multiple sessions, this is a
        list

    meta : dict
        Optional dict containing whatever you want

    n_elecs : int
        Number of electrodes

    n_secs : float
        Amount of data in seconds

    n_sessions : int
        Number of sessions

    session_labels : list
        Label for each session

    kurtosis : list of floats
        1 by number of electrode list containing kurtosis for each electrode

    Methods
    ----------

    get_data : function
        Takes brain object and returns data

    remove_elecs : function
        Takes brain object and returns brain object with electrodes and locations
        exceeding some threshold removed

    save : function
        Saves brain object

    Returns
    ----------

    brain : Brain data object
        Instance of Brain data object containing subject data

    """

    def __init__(self, data=None, locs=None, sessions=None, sample_rate=None, meta=None):

        # convert data to df
        self.data = pd.DataFrame(data)

        # locs
        self.locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

        # session
        if isinstance(sessions, str) or isinstance(sessions, int):
            self.sessions = pd.Series([session for i in range(self.data.shape[0])])
        else:
            self.sessions = pd.Series(sessions)

        # sample rate
        if isinstance(sample_rate, list):
            self.sample_rate = sample_rate
        elif isinstance(sessions, list):
            self.sample_rate = [sample_rate for s in range(sessions)]
        else:
            self.sample_rate = [sample_rate]

        # meta
        self.meta = meta

        # compute attrs
        self.n_elecs = self.data.shape[1]
        self.n_secs = self.data.shape[0]/self.sample_rate[0][0][0]
        self.date_created = time.strftime("%c")

    # methods

    def info(self):
        """
        Print info about the brain object
        """
        print('Number of electrodes: ' + str(self.n_elecs))
        print('Recording time in seconds: ' + str(self.n_secs))
        print('Number of sessions: ' + str(1))
        print('Date created: ' + str(self.date_created))
        print('Meta data: ' + str(self.meta))

    def get_data(self):
        """
        Gets data from brain object
        """
        return self.data.as_matrix()

    def remove_elecs(self, measure='kurtosis', threshold=10):
        """
        Gets data from brain object
        """
        if measure is 'kurtosis':
            pass

    def save(self, filepath):
        """
        Save a pickled brain, mwahahaha
        """
        with open(filepath + '.bo', 'wb') as f:
            pickle.dump(self, f)
            print('pickle saved.')
