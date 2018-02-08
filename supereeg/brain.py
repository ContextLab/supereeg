from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
import time
import os
import warnings
import copy
import numpy as np
import pandas as pd
import nibabel as nib
import deepdish as dd
import matplotlib.pyplot as plt
from nilearn import plotting as ni_plt
from .helpers import _kurt_vals, zscore, _normalize_Y, _vox_size

class Brain(object):
    """
    Brain data object for the supereeg package

    A brain data object contains a single iEEG subject. To create one, at minimum
    you need data (samples by electrodes), location coordinates in MNI space and
    the sample rate of the data. Additionally, you can include a session id. If
    included, all analyses will be performed within session and then aggregated
    across sessions.  You can also include a meta dict, which can contain any
    other information that might be useful (subject id, recording params, etc).

    Parameters
    ----------

    data : numpy.ndarray or pandas.DataFrame
        Samples x electrodes array containing the EEG data

    locs : numpy.ndarray or pandas.DataFrame
        MNI coordinate (x,y,z) by electrode array containing electrode locations

    session : numpy.ndarray
        Samples x 1 array containing session identifiers

    sample_rates : float or list of floats
        Sample rate of the data. If different over multiple sessions, this is a
        list

    meta : dict
        Optional dict containing whatever you want

    date created : str
        Time created

    label : list
        List delineating if location was reconstructed or observed. This is computed in reconstruction.

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
        Amount of data in seconds for each session

    n_sessions : int
        Number of sessions

    session_labels : list
        Label for each session

    kurtosis : list of floats
        1 by number of electrode list containing kurtosis for each electrode


    Returns
    ----------

    bo : supereeg.Brain instance
        Instance of Brain data object containing subject data

    """

    def __init__(self, data=None, locs=None, sessions=None, sample_rate=None,
                 meta=None, date_created=None, label=None):

        # convert data to df if not imported as df
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame(data)

        # convert locs to df if not imported as df
        if isinstance(locs, pd.DataFrame):
            self.locs = locs
        else:
            self.locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

        # session
        if isinstance(sessions, str) or isinstance(sessions, int):
            self.sessions = pd.Series([sessions for i in range(self.data.shape[0])])

        elif sessions is None:
            self.sessions = pd.Series([1 for i in range(self.data.shape[0])])
        else:
            self.sessions = pd.Series(sessions.ravel())

        # sample rate
        if isinstance(sample_rate, np.ndarray):
            if np.shape(sample_rate)[1]>1:
                self.sample_rate = list(sample_rate[0])
            elif np.shape(sample_rate)[1] == 1:
                self.sample_rate = [sample_rate[0]]
            assert len(self.sample_rate) ==  len(self.sessions.unique()), 'Should be one sample rate for each session.'

        elif isinstance(sample_rate, list):
            if isinstance(sample_rate[0], np.ndarray):
                self.sample_rate = list(sample_rate[0][0])
            else:
                self.sample_rate = sample_rate
            assert len(self.sample_rate) ==  len(self.sessions.unique()), 'Should be one sample rate for each session.'

        elif sample_rate is None:
            self.sample_rate = None
            self.n_secs = None
            warnings.warn('No sample rate given.  Number of seconds cant be computed')

        elif type(sample_rate) in [int, float]:
            self.sample_rate = [sample_rate]*len(self.sessions.unique())
        else:
            self.sample_rate = None
            warnings.warn('Format of sample rate not recognized. Number of seconds cannot be computed.'
                          'Setting sample rate to None')

        if sample_rate is not None:
            index, counts = np.unique(self.sessions, return_counts=True)
            self.n_secs = np.true_divide(counts, np.array(sample_rate))

        # meta
        if meta:
            self.meta = meta
        else:
            self.meta = {}

        if not date_created:
            self.date_created = time.strftime("%c")
        else:
            self.date_created = date_created

        # compute attrs
        self.n_elecs = self.data.shape[1] # needs to be calculated by sessions
        self.n_sessions = len(self.sessions.unique())

        # add kurtosis
        self.kurtosis = _kurt_vals(self)

        if not label:
            self.label = len(self.locs) * ['observed']
        else:
            self.label = label

    def info(self):
        """
        Print info about the brain object

        Prints the number of electrodes, recording time, number of recording
        sessions, date created, and any optional meta data.
        """
        print('Number of electrodes: ' + str(self.n_elecs))
        print('Recording time in seconds: ' + str(self.n_secs))
        print('Number of sessions: ' + str(self.n_sessions))
        print('Date created: ' + str(self.date_created))
        print('Meta data: ' + str(self.meta))

    def get_data(self):
        """
        Gets data from brain object
        """
        return self.data.as_matrix()

    def get_zscore_data(self):
        """
        Gets zscored data from brain object
        """
        return zscore(self.data.as_matrix())

    def get_locs(self):
        """
        Gets locations from brain object
        """
        return self.locs.as_matrix()

    def get_slice(self, times=None, locs=None):
        """
        Gets a time slice of the data
        """
        if not times:
            times = list(range(self.data.shape[0]))
        if not locs:
            locs = list(range(self.locs.shape[0]))

        if times and locs:
            data = self.data.iloc[times, locs].copy()
            sessions = self.sessions.iloc[times].copy()
            if self.sample_rate:
                sample_rate = [self.sample_rate[int(s-1)] for s in sessions.unique()]
            else:
                sample_rate = self.sample_rate
            meta = copy.copy(self.meta)
            locs = self.locs.iloc[locs].copy()
            date_created = self.date_created
            return Brain(data=data, locs=locs, sessions=sessions,
                         sample_rate=sample_rate, meta=meta,
                         date_created=date_created)
        else:
            return self.copy()


    def plot_data(self, filepath=None, time_min=None, time_max=None, title=None, electrode=None, threshold=10,
                  filtered=True):
        """
        Normalizes and plots data from brain object


        Parameters
        ----------
        filepath : str
            A name for the file.  If the file extension (.png) is not specified, it will be appended.

        time_min : int
            Minimum value for desired time window

        time_max : int
            Maximum value for desired time window

        title : str
            Title for plot

        electrode : int
            Location in MNI coordinate (x,y,z) by electrode df containing electrode locations
            ## should add functionality that matches the coordinate instead of location in matrix

        threshold : int
        Value of kurtosis threshold

        filtered : True
        Default to filter by kurtosis threshold.  If False, will show all original data.

        """


        # normalizes the samples x electrodes array containing the EEG data and adds 1 to each row
        # so that the y-axis value corresponds to electrode location in the MNI coordinate (x,y,z)
        # by electrode df containing electrode locations
        Y = _normalize_Y(self.data)

        # if filtered in passed, filter by electrodes that do not pass kurtosis thresholding
        if filtered:
            thresh_bool = self.kurtosis > threshold
            Y = Y.iloc[:, ~thresh_bool]

        # if electrode is included, index data at electrode location
        if electrode is not None:
            Y = Y.columns[int(electrode)]

        # divide index by sample rate so that index corresponds to time
        if self.sample_rate:
            Y.index = np.divide(Y.index,np.mean(self.sample_rate))

        # if a time window is designated index data in that window
        if all([time_min, time_max]):
            mask = (Y.index >= time_min) & (Y.index <= time_max)
            Y = Y[mask]

        # if a time window is not designated, default to the first 500 seconds
        else:
            time_min = 0
            time_max =  10
            mask = (Y.index >= time_min) & (Y.index <= time_max)
            Y= Y[mask]

        # plot data
        ax = Y.plot(legend=False, title=title, color='k', lw=.6)
        ax.set_facecolor('w')
        ax.set_xlabel("time")
        ax.set_ylabel("electrode")
        ax.set_ylim([0,len(Y.columns) + 1])
        if filepath:
            plt.savefig(filename = filepath)
        else:
            plt.show()

    def plot_locs(self, pdfpath = None):
        """
        Plots electrode locations from brain object


        Parameters
        ----------

        pdfpath : str
        A name for the file.  If the file extension (.pdf) is not specified, it will be appended.


        """
        ni_plt.plot_connectome(np.eye(self.locs.shape[0]), self.locs, display_mode='lyrz', output_file=pdfpath,
                               node_kwargs={'alpha': 0.5, 'edgecolors': None}, node_size=10,
                               node_color=np.ones(self.locs.shape[0]))
        if not pdfpath:
            ni_plt.show()

    def save(self, fname, compression='blosc'):
        """
        Save method for the brain object

        The data will be saved as a 'bo' file, which is a dictionary containing
        the elements of a brain object saved in the hd5 format using
        `deepdish`.

        Parameters
        ----------

        fname : str
            A name for the file.  If the file extension (.bo) is not specified,
            it will be appended.

        compression : str
            The kind of compression to use.  See the deepdish documentation for
            options: http://deepdish.readthedocs.io/en/latest/api_io.html#deepdish.io.save

        """


        # put bo vars into a dict
        bo = {
            'data' : self.data.as_matrix(),
            'locs' : self.locs.as_matrix(),
            'sessions' : self.sessions,
            'sample_rate' : self.sample_rate,
            'meta' : self.meta,
            'date_created' : self.date_created
        }

        # if extension wasn't included, add it
        if fname[-3:]!='.bo':
            fname+='.bo'

        # save
        dd.io.save(fname, bo, compression=compression)


    def to_nii(self, filepath=None, template=None):

        """
        Save brain object as a nifti file


        Parameters
        ----------

        filepath : str
            Path to save the nifti file

        template : str
            Path to template nifti file

        Returns
        ----------

        nifti : nibabel.Nifti1Image
            A nibabel nifti image

        """

        # recontructed voxel size:
        recon_v_size = _vox_size(self.locs)

        if template is None:

            if int(recon_v_size[0][0]) not in [20, 8, 6]:
                warnings.warn('Template is None.  Default to using a template with 20mm voxels.')
                template = os.path.dirname(os.path.abspath(__file__)) + '/data/gray_mask_20mm_brain.nii'

            elif int(recon_v_size[0][0]) in [20, 8, 6]:
                warnings.warn('Template is None.  '
                              'Default to b using a template with ' + str(int(recon_v_size[0][0])) + ' voxels.')
                template = os.path.dirname(os.path.abspath(__file__)) + \
                           '/data/gray_mask_'+ str(int(recon_v_size[0][0]))+'mm_brain.nii'

        elif int(recon_v_size[0][0]) in [20, 8, 6]:
                warnings.warn(
                    'Voxel sizes of reconstruction and template do not match. '
                    'Try '+'/data/gray_mask_'+ str(int(recon_v_size[0][0]))+'mm_brain.nii'+ ' to match voxel sizes.')

        # load template
        img = nib.load(template)
        hdr = img.get_header()

        # template voxel size:
        temp_v_size = hdr.get_zooms()[0:3]

        if not np.array_equiv(temp_v_size, recon_v_size.ravel()):
            warnings.warn('Voxel sizes of reconstruction and template do not match. '
                          'Voxel sizes calculated from model locations.')

        R = self.get_locs()
        Y = self.data.as_matrix()
        Y = np.array(Y, ndmin=2)
        S = img.affine
        locs = np.array(np.dot(R - S[:3, 3], np.linalg.inv(S[0:3, 0:3])), dtype='int')

        shape = np.max(np.vstack([np.max(locs, axis=0) + 1, img.shape[0:3]]), axis=0)
        data = np.zeros(tuple(list(shape) + [Y.shape[0]]))
        counts = np.zeros(data.shape)

        # loop over data and locations to fill in activations
        for i in range(Y.shape[0]):
            for j in range(R.shape[0]):
                data[locs[j, 0], locs[j, 1], locs[j, 2], i] += Y[i, j]
                counts[locs[j, 0], locs[j, 1], locs[j, 2], i] += 1
        with np.errstate(invalid='ignore'):
            data = np.divide(data, counts)
        data[np.isnan(data)] = 0
        nifti =  nib.Nifti1Image(data, affine=img.affine)

        # save if filepath
        if filepath:
            nifti.to_filename(filepath)

        return nifti
