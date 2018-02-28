from __future__ import division
from __future__ import print_function

import nibabel as nib
from nibabel import Nifti1Image
import numpy as np
from .helpers import _nifti_to_brain

import six # Python 2 and 3 compatibility

class Nifti(Nifti1Image):
    """
    Nifti class for the supereeg package

    Child class for nifti parent class

    Parameters
    ----------

    data : Nifti1Image or supereeg.Brain, supereeg.Nifti

        If data is a nifti image (either supereeg.Nifti or Nifti1Image), returns nifti values.


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


    def nii_to_brain(self):

        from .brain import Brain

        return Brain(_nifti_to_brain(self))

