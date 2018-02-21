# -*- coding: utf-8 -*-
"""
=============================
Load and save nifti file
=============================

This example loads a nifti file and converts it into a brain object.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se
from nilearn import plotting as ni_plt
from supereeg.helpers import _std, _gray
from supereeg.load import get_brain_object
import nibabel as nib
import os
import hypertools as hyp


# load nifti -> brain object
bo = se.load('example_data')

# export brain object -> nifti
nifti_a = bo.to_nii()
ni_plt.plot_glass_brain(nifti_a)

# export brain object -> nifti
nifti_b = bo.to_nii(vox_size=20)
ni_plt.plot_glass_brain(nifti_b)

nifti_c = bo.to_nii(vox_size=[20, 20,20])
ni_plt.plot_glass_brain(nifti_c)

nifti_d = bo.to_nii(template=_gray([20, 20, 20]))
ni_plt.plot_glass_brain(nifti_d)

# plot the result
ni_plt.plot_anat(nifti_a)
ni_plt.show()