from __future__ import division
from __future__ import print_function


from nibabel import Nifti1Image
from nilearn import plotting as ni_plt
import numpy as np
import six # Python 2 and 3 compatibility

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

    def to_bo(self):

        from .brain import Brain

        return Brain(self)

    def to_mo(self):

        from .brain import Brain
        from .model import Model

        bo = Brain(self)
        return Model(bo)

    def plot_anat(self, pdfpath=None):

        ni_plt.plot_anat(self)
        if not pdfpath:
            ni_plt.show()

    def plot_glass_brain(self, pdfpath=None):

        ni_plt.plot_glass_brain(self)
        if not pdfpath:
            ni_plt.show()
