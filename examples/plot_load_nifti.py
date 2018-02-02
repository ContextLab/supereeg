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
from nilearn import plotting as ni_plt

# load nifti -> brain object
bo = se.load('gray_mask_6mm_brain')

# export brain object -> nifti
nifti = bo.to_nii()

# plot the result
ni_plt.plot_anat(nifti)
ni_plt.show()
