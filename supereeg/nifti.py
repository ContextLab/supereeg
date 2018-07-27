from __future__ import division
from __future__ import print_function

import numpy as np
import six # Python 2 and 3 compatibility
from nibabel import Nifti1Image
from nilearn.image import concat_imgs, index_img
from nilearn import plotting as ni_plt
from .helpers import make_gif_pngs
from .brain import Brain

class Nifti(Nifti1Image):
    """
    Nifti class for the supereeg package.  Extends the Nibabel.Nifti1Image class. #TODO: add note on documentation strangeness with reference to nilearn API

    Parameters
    ----------

    data : object or path to Nifti1Image, supereeg.Brain, supereeg.Model, supereeg.Nifti or np.ndarray

        Data can be a nifti image (either supereeg.Nifti or path to Nifti1Image), supereeg.Brain object,
           supereeg.Model object, or a np.ndarray an N-D array containing the image data


    affine : np.ndarray

        A (4, 4) affine matrix mapping array coordinates to coordinates in MNI coordinate space.


    header : nibabel.nifti1.Nifti1Header
        Image metadata in the form of a header (optional parameter, use when creating nifti object from np.ndarray).


    Returns
    ----------

    nii : supereeg.Nifti
        Instance of Nifti data class, (nibabel.nifti1.Nifti1Image subclass)

    """


    def __init__(self, data, affine=None, **kwargs):

        from .load import load, datadict
        from .brain import Brain
        from .model import Model

        if isinstance(data, six.string_types):
            if data in datadict.keys():
                data = load(data)
            else:
                image = Nifti1Image.load(data)
                super(Nifti, self).__init__(image.dataobj, image.affine)

        if isinstance(data, np.ndarray):
            if affine is None:
                raise IOError("If data is provided as array, affine must also be provided")
            else:
                super(Nifti,self).__init__(data,affine,**kwargs)

        elif isinstance(data, Nifti):
            super(Nifti, self).__init__(data.dataobj, data.affine)

        elif isinstance(data, Brain):
            data = data.to_nii(**kwargs)
            super(Nifti, self).__init__(data.dataobj, data.affine)

        elif isinstance(data, Model):
            bo = Brain(data)
            data = bo.to_nii(**kwargs)
            super(Nifti, self).__init__(data.dataobj, data.affine)

    def info(self):
        """
        Print info about the Nifti

        Prints the header information
        Could print more if necessary

        """
        print('Header: ' + str(self.header))


    def get_slice(self, index):

        if len(self.shape)>3:

            if self.shape[3]>1:

                cat_nii = []
                if hasattr(type(index), "__iter__"):
                    for i in index:
                        nii = index_img(self, i)
                        cat_nii.append(nii)

                    return concat_imgs(cat_nii)


    def plot_anat(self, pdfpath=None, index=1):

        """
        Plots nifti data

        Parameters
        ----------
        nifti : nifti image
            Nifti image to plot

        pdfpath : str or None
            Path to save pdf

        index : int or list
            Timepoint to plot

        Returns
        ----------
        results: nilearn plot_anat
            plot data


        """
        if len(self.shape)>3:

            if self.shape[3]>1:

                if hasattr(type(index), "__iter__"):
                    for i in index:
                        nii = index_img(self, i)
                        ni_plt.plot_anat(nii)
                else:
                    nii = index_img(self, index)
                    ni_plt.plot_anat(nii)

            else:
                ni_plt.plot_anat(self)
        else:
            ni_plt.plot_anat(self)

        if not pdfpath:
            ni_plt.show()


    def plot_glass_brain(self, pdfpath=None, index=1):

        """
        Plots nifti data

        Parameters
        ----------
        nifti : nifti image
            Nifti image to plot

        pdfpath : str or None
            Path to save pdf

        index : int or list
            Timepoint to plot

        Returns
        ----------
        results: nilearn plot_glass_brain
            plot data


        """
        if len(self.shape)>3:

            if self.shape[3] > 1:

                if hasattr(type(index), "__iter__"):
                    for i in index:
                        nii = index_img(self, i)
                        ni_plt.plot_glass_brain(nii)
                else:
                    nii = index_img(self, index)
                    ni_plt.plot_glass_brain(nii)
            else:
                ni_plt.plot_glass_brain(self)

        if not pdfpath:
            ni_plt.show()

    def make_gif(self, gifpath, index=range(0, 10), name=None, **kwargs):

        """
        Plots nifti data as png and compiles as gif

        Parameters
        ----------
        nifti : nifti image
            Nifti image to plot

        gifpath : str
            Path to save pngs (necessary argument)

        index : int or list
            Timepoints to plot

        name : str
            Name for png files

        Returns
        ----------
        results: nilearn plot_glass_brain gif
            plot data and compile gif


        """
        assert len(self.shape)>3, '4D necessary for gif'

        make_gif_pngs(self, gifpath, index, name, **kwargs)


    def get_locs(self):
        """
        Return locations of voxels
        """
        bo = Brain(self)
        return bo.get_locs()


    def save(self, filepath):
        """
        Save file to disk
        """
        self.to_filename(filepath)
