# -*- coding: utf-8 -*-
"""
=============================
Load and save nifti file
=============================

This example loads a nifti file and converts it into a brain object.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

# import
import superEEG as se
from nilearn import plotting

# load nifti -> brain object
bo = se.load('/Users/andyheusser/Documents/github/superEEG/superEEG/data/gray_mask_8mm_brain.nii')

# export brain object -> nifti
nifti = bo.to_nii()

# plot the result
plotting.plot_anat(nifti)
plotting.show()
