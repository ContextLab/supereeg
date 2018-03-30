# -*- coding: utf-8 -*-
"""
=============================
Plot Nifti
=============================

Here, we load an example nifti image and plot it two ways.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se

# load nifti objects by passing directly to Nifti class
# if no parameters are passed default uses gray matter masked brain downsampled to 6 mm resolution
bo_nii = se.Nifti('example_data', vox_size=6)

# plot first 2 timepoints as plot_glass_brain
# default will plot first timepoint
bo_nii.plot_glass_brain(index=[0,1])



