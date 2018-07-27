from __future__ import division
from __future__ import print_function
import time
import os
import warnings
import copy
import six
import numpy as np
import pandas as pd
import nibabel as nib
import deepdish as dd
import matplotlib.pyplot as plt

from .helpers import _kurt_vals, _normalize_Y, _vox_size, _resample, _plot_locs_connectome, \
    _plot_locs_hyp, _std, _gray, _nifti_to_brain, _brain_to_nifti, _z_score

class Brain(object):
    """
    Brain data object for the supereeg package

    A brain data object contains a single iEEG subject. To create one, at minimum
    you need data (samples x electrodes), location coordinates in MNI space and
    the sample rate of the data. Additionally, you can include a session id. If
    included, all analyses will be performed within session and then aggregated
    across sessions.  You can also include a meta dict, which can contain any
    other information that might be useful (subject id, recording params, etc).

    Parameters
    ----------

    data : numpy.ndarray or pandas.DataFrame, supereeg.Model, supereeg.Nifti, or Nifti1Image

        Samples x electrodes array containing the iEEG data.

        If data is a model, returns correlation matrix.

        If data is a nifti image (either supereeg.Nifti or Nifti1Image), returns nifti values as samples by electrodes
        array.

    locs : numpy.ndarray or pandas.DataFrame
        Electrode by MNI coordinate (x,y,z) array containing electrode locations

    session : str, int or numpy.ndarray
        Samples x 1 array containing session identifiers for each time sample.
        If str or int, the value will be copied for each time sample.

    sample_rates : float, int or list
        Sample rate (Hz) of the data. If different over multiple sessions, this is a list.

    meta : dict
        Optional dict containing whatever you want.

    date created : str
        Time created (optional)

    label : list
        List delineating if location was reconstructed or observed. This is computed in reconstruction.

    Attributes
    ----------

    data : pandas.DataFrame
        Samples x electrodes dataframe containing the EEG data.

    locs : pandas.DataFrame
        Electrode by MNI coordinate (x,y,z) df containing electrode locations.

    sessions : pandas.Series
        Samples x 1 array containing session identifiers.  If a single value is passed, a single session will be
        created.

    sample_rates : list
        Sample rate of the data. If different over multiple sessions, this is a list.

    meta : dict
        Optional dict containing whatever you want.

    n_elecs : int
        Number of electrodes

    dur : float
        Amount of data in seconds for each session

    n_sessions : int
        Number of sessions

    label : list
        Label for each session

    kurtosis : int
        Kurtosis threshold

    filter : 'kurtosis' or None
        If 'kurtosis', electrodes that exceed the kurtosis threshold will be removed.  If None, no thresholding is
        applied.

    minimum_voxel_size : positive scalar or 3D numpy array
        Used to construct Nifti objects; default: 3 (mm)

    maximum_voxel_size : positive scalar or 3D numpy array
        Used to construct Nifti objects; default: 20 (mm)


    Returns
    ----------

    bo : supereeg.Brain
        Instance of Brain data object.

    """

    def __init__(self, data=None, locs=None, sessions=None, sample_rate=None,
                 meta=None, date_created=None, label=None, kurtosis=None,
                 kurtosis_threshold=10, minimum_voxel_size=3, maximum_voxel_size=20,
                 filter='kurtosis'):

        from .load import load
        from .model import Model
        from .nifti import Nifti

        if isinstance(data, six.string_types):
            data = Brain(load(data))

        if isinstance(data, Brain):
            self.__dict__.update(data.__dict__)
            self.update_filter_inds()
            self.update_info()
            self = data

        else:
            if isinstance(data, (Nifti, nib.nifti1.Nifti1Image)):
                warnings.simplefilter('ignore')
                data, locs, meta = _nifti_to_brain(data)
                sample_rate = 1

            if isinstance(data, Model):
                locs = data.locs
                data = data.get_model(z_transform=False)

            if isinstance(data, pd.DataFrame):
                self.data = data
            else:
                self.data = pd.DataFrame(data)

            if isinstance(locs, pd.DataFrame):
                assert all(locs.columns == ['x', 'y', 'z'])
                self.locs = locs
            else:
                self.locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

            if isinstance(sessions, str) or isinstance(sessions, int):
                self.sessions = pd.Series([sessions for i in range(self.data.shape[0])])

            elif sessions is None:
                self.sessions = pd.Series([1 for i in range(self.data.shape[0])])
            else:
                self.sessions = pd.Series(sessions.ravel())

            if type(sample_rate) in [int, float]:
                self.sample_rate = [sample_rate]*len(self.sessions.unique())
            elif isinstance(sample_rate, list):
                if isinstance(sample_rate[0], np.ndarray):
                    if sample_rate[0].ndim == 1:
                        sample_rate = np.atleast_2d(sample_rate)
                        self.sample_rate = [sample_rate[0]]
                    else:
                        self.sample_rate = list(sample_rate[0][0])
                else:
                    self.sample_rate = sample_rate
            elif isinstance(sample_rate, np.ndarray):
                if sample_rate.ndim == 1:
                    sample_rate = np.atleast_2d(sample_rate)
                if np.shape(sample_rate)[1]>1:
                    self.sample_rate = list(sample_rate[0])
                elif np.shape(sample_rate)[1] == 1:
                    self.sample_rate = [sample_rate[0]]
                assert len(self.sample_rate) ==  len(self.sessions.unique()), \
                    'Should be one sample rate for each session.'
            else:
                self.sample_rate = None

                if self.data.shape[0] == 1:
                    self.dur = 0
                else:
                    self.dur = None
                    warnings.warn('No sample rate given.  Number of seconds cant be computed')

            if sample_rate is not None:
                index, counts = np.unique(self.sessions, return_counts=True)
                self.dur = np.true_divide(counts, np.array(sample_rate))

            if meta:
                self.meta = meta
            else:
                self.meta = {}

            if not date_created:
                self.date_created = time.strftime("%c")
            else:
                self.date_created = date_created

            self.n_elecs = self.data.shape[1] # needs to be calculated by sessions
            self.n_sessions = len(self.sessions.unique())
            if np.iterable(kurtosis):
                self.kurtosis = kurtosis
            else:
                self.kurtosis = _kurt_vals(self)
            self.kurtosis_threshold = kurtosis_threshold
            self.filter=filter
            self.filter_inds = self.update_filter_inds()

            if not label:
                self.label = len(self.locs) * ['observed']
            else:
                self.label = label

            self.minimum_voxel_size = minimum_voxel_size
            self.maximum_voxel_size = maximum_voxel_size

    def __getitem__(self, slice):
        if isinstance(slice, tuple):
            timeslice, locslice = slice
        else:
            timeslice = slice
            locslice = None
        return self.get_slice(sample_inds=timeslice, loc_inds=locslice)

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter >= self.data.shape[0]:
            raise StopIteration
        s = self[self.counter]
        self.counter+=1
        return s

    def next(self):
        """
        Return next sample from Brain object (wrapper for self.__next__)
        """
        return self.__next__()

    def update_filter_inds(self):
        if self.filter == 'kurtosis':
            self.filter_inds = self.kurtosis <= self.kurtosis_threshold
        else:
            self.filter_inds = np.ones((1, self.locs.shape[0]), dtype=np.bool)[0] #TODO: check this

    def update_info(self):
        self.n_elecs = self.data.shape[1] # needs to be calculated by sessions
        self.n_sessions = len(self.sessions.unique())
        ## not entirely sure if try/except necessary and not if/else
        try:
            index, counts = np.unique(self.sessions, return_counts=True)
            self.dur = np.true_divide(counts, np.array(self.sample_rate))
        except:
            self.dur = None

    def info(self):
        """
        Print info about the brain object

        Prints the number of electrodes, recording time, number of recording
        sessions, date created, and any optional meta data.
        """
        self.update_info()
        print('Number of electrodes: ' + str(self.n_elecs))
        print('Recording time in seconds: ' + str(self.dur))
        print('Sample Rate in Hz: '+ str(self.sample_rate))
        print('Number of sessions: ' + str(self.n_sessions))
        print('Date created: ' + str(self.date_created))
        print('Meta data: ' + str(self.meta))

    def apply_filter(self, inplace=True):
        """ Return a filtered copy """

        if self.filter is None:
            if not inplace:
                return copy.deepcopy(self)
            else:
                return None

        x = copy.copy(self.__dict__)
        x['data'] = self.get_data()
        x['locs'] = self.get_locs()

        if self.filter == 'kurtosis':
            x['kurtosis'] = x['kurtosis'][x['kurtosis'] <= x['kurtosis_threshold']]

        for key in ['n_subs', 'n_elecs', 'n_sessions', 'dur', 'filter_inds']:
            if key in x.keys():
                x.pop(key)

        boc = Brain(**x)
        boc.filter = None
        boc.update_info()
        if inplace:
            self.__init__(boc)
        else:
            return boc

    def get_data(self):
        """
        Gets data from brain object
        """
        self.update_filter_inds()
        return self.data.iloc[:, self.filter_inds.ravel()].reset_index(drop=True)

    def get_zscore_data(self):
        """
        Gets zscored data from brain object
        """
        self.update_filter_inds()
        return _z_score(self)

    def get_locs(self):
        """
        Gets locations from brain object
        """
        self.update_filter_inds()
        return self.locs.iloc[self.filter_inds.ravel(), :].reset_index(drop=True)

    def get_slice(self, sample_inds=None, loc_inds=None, inplace=False):
        """
        Indexes brain object data

        Parameters
        ----------
        sample_inds : int or list
            Times you wish to index

        loc_inds : int or list
            Locations you with to index

        inplace : bool
            If True, indexes in place.

        """
        if sample_inds is None:
            sample_inds = list(self.get_data().index)
        if loc_inds is None:
            loc_inds = list(self.get_locs().index)
        if isinstance(sample_inds, int):
            sample_inds = [sample_inds]
        if isinstance(loc_inds, int):
            loc_inds = [loc_inds]

        data = self.get_data().iloc[sample_inds, loc_inds].reset_index(drop=True)
        sessions = self.sessions.iloc[sample_inds]
        kurtosis = self.kurtosis[self.get_locs().index[loc_inds]]
        if self.sample_rate:
            sample_rate = [self.sample_rate[int(s-1)] for s in
                           sessions.unique()]
        else:
            sample_rate = self.sample_rate
        meta = copy.copy(self.meta)
        locs = self.get_locs().iloc[loc_inds].reset_index(drop=True)
        date_created = time.strftime("%c")

        b = Brain(data=data, locs=locs, sessions=sessions, sample_rate=sample_rate, meta=meta, date_created=date_created,
                  filter=None, kurtosis=kurtosis)
        if inplace:
            self = b
        else:
            return b

    def resample(self, resample_rate=None):
        """
        Resamples data


        Parameters
        ----------
        resample_rate : int or float
            Desired sample rate

        """
        if resample_rate is None:
            return self
        else:
            data, sessions, sample_rate = _resample(self, resample_rate)
            self.data = data
            self.sessions = sessions
            self.sample_rate = sample_rate


    def plot_data(self, filepath=None, time_min=None, time_max=None, title=None,
                  electrode=None):
        """
        Normalizes and plots data from brain object


        Parameters
        ----------
        filepath : str
            A name for the file.  If the file extension (.png) is not specified,
            it will be appended.

        time_min : int
            Minimum value for desired time window

        time_max : int
            Maximum value for desired time window

        title : str
            Title for plot

        electrode : int
            Location in MNI coordinate (x,y,z) by electrode df containing electrode locations
        """

        # normalizes the samples x electrodes array containing the EEG data and
        # adds 1 to each row so that the y-axis value corresponds to electrode
        # location in the MNI coordinate (x,y,z) by electrode df containing
        # electrode locations

        if self.get_data().shape[0] == 1:
            nii = self.to_nii()
            nii.plot_glass_brain()
        elif self.get_data().empty:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            ax.set_facecolor('w')
            ax.set_xlabel("time")
            ax.set_ylabel("electrode")
            if filepath:
                plt.savefig(filename=filepath)
            else:
                plt.show()
        else:
            Y = _normalize_Y(self.get_data())

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

            ax = Y.plot(legend=False, title=title, color='k', lw=.6)
            ax.set_facecolor('w')
            ax.set_xlabel("time")
            ax.set_ylabel("electrode")
            ax.set_ylim([0, len(Y.columns) + 1])
            if filepath:
                plt.savefig(filename=filepath)
            else:
                plt.show()


    def plot_locs(self, pdfpath=None):
        """
        Plots electrode locations from brain object

        Colors:
            - Observed : Blue
            - Removed : Cyan
            - Reconstructed : Red

        Parameters
        ----------
        pdfpath : str
            A name for the file.  If the file extension (.pdf) is not specified, it will be appended.

        """


        locs = self.locs

        if self.filter_inds is None:
            label = np.array(self.label)
        elif self.filter_inds.all():
            label = np.array(self.label)
        else:
            label = np.array(list(map(lambda x: 'observed' if x else 'removed', self.filter_inds)))
        if locs.shape[0] <= 10000:
            _plot_locs_connectome(locs, label, pdfpath)
        else:
            _plot_locs_hyp(locs, pdfpath)

    def to_nii(self, filepath=None, template='gray', vox_size=None, sample_rate=None):

        """
        Save brain object as a nifti file.


        Parameters
        ----------

        filepath : str

            Path to save the nifti file

        template : str, Nifti1Image, or None

            Template is a nifti file with the desired resolution to save the brain object activity

                If template is None (default) :
                    - Uses gray matter masked brain downsampled to brain object voxel size (max 20 mm)

                If template is str :
                    - Checks if nifti file path and uses specified nifti

                    - If not a filepath, checks if 'std' or 'gray'
                        - 'std': Uses standard brain downsampled to brain object voxel size
                        - 'gray': Uses gray matter masked brain downsampled to brain object voxel size

                If template is Nifti1Image :
                    - Uses specified Nifti image

        Returns
        ----------

        nifti : supereeg.Nifti
            A supereeg nifti object

        """
        from .nifti import Nifti

        if vox_size:
            v_size = vox_size
        else:
            v_size = _vox_size(self.locs)

        if np.isscalar(self.minimum_voxel_size):
            mnv = np.multiply(self.minimum_voxel_size, np.ones_like(v_size))
        else:
            mnv = self.minimum_voxel_size

        if np.isscalar(self.maximum_voxel_size):
            mxv = np.multiply(self.maximum_voxel_size, np.ones_like(v_size))
        else:
            mxv = self.maximum_voxel_size

        if np.any(v_size < self.minimum_voxel_size):
            v_size[v_size < self.minimum_voxel_size] = mnv[v_size < self.minimum_voxel_size]

        if np.any(v_size > self.maximum_voxel_size):
            v_size[v_size > self.maximum_voxel_size] = mxv[v_size > self.maximum_voxel_size]

        if template is None:
            img = _gray(v_size)

        elif type(template) is nib.nifti1.Nifti1Image:
            img = template

        elif isinstance(template, str) or isinstance(template, basestring):

            if os.path.exists(template):
                img = nib.load(template)

            elif template is 'gray':
                img = _gray(v_size)

            elif template is 'std':
                img = _std(v_size)

            else:
                warnings.warn('template format not supported')
        else:
            warnings.warn('Nifti format not supported')

        if sample_rate:
            data, sessions, sample_rate = _resample(self, sample_rate)
            self.data = data
            self.sessions = sessions
            self.sample_rate = sample_rate


        hdr = img.get_header()
        temp_v_size = hdr.get_zooms()[0:3]

        if not np.array_equiv(temp_v_size, v_size):
            warnings.warn('Voxel sizes of reconstruction and template do not match. '
                          'Voxel sizes calculated from model locations.')

        nifti = _brain_to_nifti(self, img)

        if filepath:
            nifti.to_filename(filepath)

        return nifti


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

        bo = {
            'data': self.data.as_matrix(),
            'locs': self.locs,
            'sessions': self.sessions,
            'sample_rate': self.sample_rate,
            'kurtosis': self.kurtosis,
            'kurtosis_threshold' : self.kurtosis_threshold,
            'meta': self.meta,
            'date_created': self.date_created,
            'minimum_voxel_size': self.minimum_voxel_size,
            'maximum_voxel_size': self.maximum_voxel_size,
            'label' : self.label,
            'filter' : self.filter,
        }

        if fname[-3:] != '.bo':
            fname += '.bo'

        dd.io.save(fname, bo, compression=compression)
