# -*- coding: utf-8 -*-

import pandas as pd
import time
import os
import numpy as np
import pickle
import nibabel as nib
from nibabel.affines import apply_affine
import warnings
from ._helpers.stats import kurt_vals

class Brain(object):
    """
    Brain data object for the superEEG package

    A brain data object contains a single iEEG subject. To create one, at minimum
    you need data (samples by electrodes), location coordinates in MNI space and
    the sample rate of the data. Additionally, you can include a session id. If
    included, all analyses will be performed within session and then aggregated
    across sessions.  You can also include a meta dict, which can contain any
    other information that might be useful (subject id, recording params, etc).

    Parameters
    ----------

    data : numpy.ndarray
        Samples x electrodes array containing the EEG data

    locs : numpy.ndarray
        MNI coordinate (x,y,z) by electrode array containing electrode locations

    session : numpy.ndarray
        Samples x 1 array containing session identifiers

    sample_rates : float or list of floats
        Sample rate of the data. If different over multiple sessions, this is a
        list

    meta : dict
        Optional dict containing whatever you want

    Attributes
    ----------

    data : Pandas DataFrame
        Samples x electrodes dataframe containing the EEG data

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


    Returns
    ----------

    bo : Brain data object
        Instance of Brain data object containing subject data

    """

    def __init__(self, data=None, locs=None, sessions=None, sample_rate=None, meta=None):

        # convert data to df
        self.data = pd.DataFrame(data)

        # locs
        self.locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

        # session
        if isinstance(sessions, str) or isinstance(sessions, int):
            self.sessions = pd.Series([sessions for i in range(self.data.shape[0])])
        elif sessions is None:
            self.sessions = pd.Series([1 for i in range(self.data.shape[0])])
        else:
            self.sessions = pd.Series(sessions.ravel())

        # sample rate
        if isinstance(sample_rate, list):
            self.sample_rate = sample_rate
        elif isinstance(sessions, list):
            self.sample_rate = [sample_rate for s in self.sessions.values]
        elif sample_rate is None:
            self.sample_rate = [1000 for s in self.sessions.values]
            warnings.warn('No sample rate given.  Setting sample rate to 1000')
        else:
            self.sample_rate = [sample_rate]

        # meta
        self.meta = meta

        # compute attrs
        self.n_elecs = self.data.shape[1]
        ## needs to be calculated by sessions
        self.n_secs = self.data.shape[0]/self.sample_rate[0]
        self.date_created = time.strftime("%c")

        # add methods
        self.kurtosis = kurt_vals(self)

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

    def to_pickle(self, filepath):
        """
        Save a pickled brain, mwahahaha
        """
        with open(filepath + '.bo', 'wb') as f:
            pickle.dump(self, f)
            print('Brain object saved as pickle.')

    def to_nifti(self, filepath=None,
                 template='../superEEG/data/MNI152_T1_6mm_brain.nii.gz'):
        """
        Save brain object as a nifti file
        """

        img = nib.load(template)

        def normalize_locs(x, vox_size):
            shifted = (x - np.min(x, axis=0)) + vox_size
            return np.round(np.divide(shifted, vox_size))-1

        data = np.zeros(tuple(list(img.shape)+[self.data.shape[0]]))
        locs = normalize_locs(self.locs, img.header.get_zooms())

        for i, row in self.data.iterrows():
            for j, loc in locs.iterrows():
                a,b,c,d = np.array(loc.values.tolist()+[i]).astype(int)
                data[a,b,c,d]=row.loc[j]

        nifti = nib.nifti1.Nifti1Image(data, img.affine, header=img.header)

        if filepath:
            nifti.to_filename(filepath)

        return nifti
