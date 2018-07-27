from __future__ import division
from __future__ import print_function
import time
import six
import numpy as np
import pandas as pd
import deepdish as dd

from .brain import Brain
from .helpers import _unique, _union, _count_overlapping, tal2mni, _plot_locs_connectome, \
    _plot_locs_hyp

class Location(object):
    """
    Location data object for the supereeg package

    A location data object contains electrode locations

    Parameters
    ----------

    data : numpy.ndarray or pandas.DataFrame, supereeg.Model, supereeg.Nifti, or Nifti1Image

        Any means of providing location data

    meta : dict
        Optional dict containing whatever you want.

    reference : 'tal' or 'mni' (default: 'mni')
        Specify whether coordinates are provided in Talairach or MNI space

    date_created : str
        Time created (optional)

    Attributes
    ----------

    locs : pandas.DataFrame
        Electrode by MNI coordinate (x,y,z) df containing electrode locations.

    meta : the meta dict

    date_created : str

    Returns
    ----------

    lo : supereeg.Location
        Instance of Location data object

    """

    def __init__(self, data=None, meta=None, date_created=None, reference='mni'):

        from .load import load
        from .model import Model
        from .nifti import Nifti

        if isinstance(data, Location):
            self = data
            return
        elif isinstance(data, six.string_types):
            self = Location(load(data))
            return
        elif (isinstance(data, Brain) or isinstance(data, Model) or isinstance(data, Nifti)):
            self = data.get_locs()
            return
        elif isinstance(data, np.array):
            assert data.shape[1] == 3, 'Locations must be 3D'
            data = pd.DataFrame(data=data, columns=['x', 'y', 'z'])
        elif isinstance(data, pd.DataFrame):
            assert 'x' in data.columns, 'Must specify labeled x-coordinate of locations'
            assert 'y' in data.columns, 'Must specify labeled y-coordinate of locations'
            assert 'z' in data.columns, 'Must specify labeled z-coordinate of locations'
            data = data[['x', 'y', 'z']] #ensure dimensions are sorted in x, y, z order

        if reference == 'tal':
            data = tal2mni(data.as_matrix())
            data = pd.DataFrame(data=data, columns=['x', 'y', 'z'])
        else:
            assert reference == 'mni', 'Locations must be specified in either Talairach or MNI space.'

        data, tmp = _unique(data)
        data.index = np.arange(data.shape[0])

        self.locs = data
        self.meta = meta

        if date_created is None:
            self.date_created = time.strftime("%c")
        else:
            self.date_created = date_created


    def __getitem__(self, i):
        return self.locs.iloc[i]

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
        return self.__next__()

    def merge(self, new_locs):
        """
        Return a new Location object containing the union of self.locs and new_locs
        """
        new_locs = Location(new_locs)

        combined_locs = _union(self.get_locs(), new_locs.get_locs())
        self.locs = combined_locs

    def is_subset(self, x):
        """
        Return True if and only if the locations in self.locs are all contained in x.locs
        """
        x = Location(x)
        return np.all(_count_overlapping(x.get_locs(), self.get_locs()))

    def is_superset(self, x):
        """
        Return True if and only if the locations in x.locs are all contained in self.locs
        """
        x = Location(x)
        return np.all(_count_overlapping(self.get_locs(), x.get_locs()))

    def info(self):
        """
        Print info about the brain object

        Prints the number of electrodes, recording time, number of recording
        sessions, date created, and any optional meta data.
        """
        self.update_info()
        print('Number of electrodes: ' + str(self.locs.shape[0]))
        print('Date created: ' + str(self.date_created))
        print('Meta data: ' + str(self.meta))

    def get_locs(self):
        """
        Gets locations from brain object
        """
        return self.locs

    def plot_locs(self, pdfpath=None):
        """
        Plots electrode locations

        Parameters
        ----------
        pdfpath : str
            A name for the file.  If the file extension (.pdf) is not specified, it will be appended.

        """

        if self.locs.shape[0] <= 10000:
            _plot_locs_connectome(self.locs, label=None, pdfpath=pdfpath)
        else:
            _plot_locs_hyp(self.locs, pdfpath)


    def save(self, fname, compression='blosc'):
        """
        Save method for the location object

        The data will be saved as a 'locs' file, which is a dictionary containing
        the elements of a locations object saved in the hd5 format using
        `deepdish`.

        Parameters
        ----------

        fname : str
            A name for the file.  If the file extension (.locs) is not specified,
            it will be appended.

        compression : str
            The kind of compression to use.  See the deepdish documentation for
            options: http://deepdish.readthedocs.io/en/latest/api_io.html#deepdish.io.save

        """

        lo = {
            'locs': self.locs,
            'meta': self.meta,
            'date_created': self.date_created,
        }

        if fname[-5:] != '.locs':
            fname += '.locs'

        dd.io.save(fname, lo, compression=compression)