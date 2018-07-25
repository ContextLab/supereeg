# -*- coding: utf-8 -*-
"""
=============================
Load and plot nifti file
=============================

This example loads a nifti file and plots it.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se

# load example nifti
# gray matter masked MNI152 brain downsampled to 20mm
nii = se.load('example_nifti')

# plot nifti
nii.plot_anat()
