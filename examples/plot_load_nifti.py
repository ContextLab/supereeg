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

# load nifti -> brain object
bo = se.load('gray_mask_20mm_brain')

# export brain object -> nifti
nifti = bo.to_nii()

# plot the result
ni_plt.plot_anat(nifti)
ni_plt.show()
