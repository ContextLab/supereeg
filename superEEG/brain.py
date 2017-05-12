# -*- coding: utf-8 -*-
import pandas as pd

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

    session : num or str
        Samples x 1 array containing session identifiers

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
        takes brain object and returns data

    remove_elecs : function
        takes brain object and returns brain object with electrodes and locations
        exceeding some threshold removed

    Returns
    ----------

    brain : Brain data object
        Instance of Brain data object containing subject data

    """

    def __init__(self):
        pass

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
            # remove kurtotic elecs and return data object
