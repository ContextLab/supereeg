from __future__ import division
from __future__ import print_function

import numpy as np
import six # Python 2 and 3 compatibility
from nibabel import Nifti1Image
from nilearn import image
from nilearn import plotting as ni_plt
from .helpers import make_gif_pngs

class Nifti(Nifti1Image):
    """
    Nifti class for the supereeg package

    Child class for nifti parent class

    Parameters
    ----------

    data : path to Nifti1Image, supereeg.Brain, supereeg.Nifti

        If data is a nifti image (either supereeg.Nifti or path to Nifti1Image), returns nifti values.


    Attributes
    ----------



    Returns
    ----------

    nii : supereeg.Nifti
        Instance of Nifti data class

    """


    def __init__(self,data,affine=None,**kwargs):

        from .brain import Brain
        from .model import Model

        if isinstance(data,six.string_types):
            image = Nifti1Image.load(data)
            super(Nifti, self).__init__(image.dataobj,image.affine)

        elif isinstance(data,np.ndarray):
            if affine is None:
                raise IOError("If data is provided as array, affine and header must also be provided")
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

    ### Ask if we want this type for method for each of the classes
    # def to_bo(self):
    #     from .brain import Brain
    #
    #     return Brain(self)
    #
    # def to_mo(self):
    #     from .brain import Brain
    #     from .model import Model
    #
    #     bo = Brain(self)
    #     return Model(bo)

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
            if hasattr(type(index), "__iter__"):
                for i in index:
                    nii = image.index_img(self, i)
                    ni_plt.plot_anat(nii)
            else:
                nii = image.index_img(self, index)
                ni_plt.plot_anat(nii)
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
            if hasattr(type(index), "__iter__"):
                for i in index:
                    nii = image.index_img(self, i)
                    ni_plt.plot_glass_brain(nii)
            else:
                nii = image.index_img(self, index)
                ni_plt.plot_glass_brain(nii)
        else:
            ni_plt.plot_glass_brain(self)

        if not pdfpath:
            ni_plt.show()

    def make_gif(self, gifpath, index=range(100, 200),name=None, **kwargs):

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


    def save(self, filepath):

        self.to_filename(filepath)
