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
import nibabel as nib
import os
# load nifti -> brain object

nii = _std(6)

nii_gray = _gray(6)
#ni_plt.plot_glass_brain(nii)

nii_b = nib.load(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/gray_mask_6mm_brain.nii')
#bo = se.load('gray_mask_6mm_brain')

# export brain object -> nifti
nifti = bo.to_nii()

# plot the result
ni_plt.plot_anat(nifti)
ni_plt.show()
