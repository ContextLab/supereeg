# -*- coding: utf-8 -*-
"""
=============================
Plot Nifti
=============================

Here, we load an example nifti image and plot it.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se

# load nifti
nii = se.load('example_nifti')

# plot anatomy
nii.plot_anat()

# plot as plot_glass_brain
nii.plot_glass_brain()



