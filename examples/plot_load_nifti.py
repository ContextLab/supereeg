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


# load nifti
# example nifti is the gray matter masked MNI152 brain downsampled to 20mm
nii = se.load('example_nifti')

# plot nifti
nii.plot_anat()

# nifti -> brain object
bo = se.Brain(nii)

# export brain object -> nifti
nii_r = bo.to_nii(template='gray', vox_size=20)

# # plot the result (same as before)
nii_r.plot_anat()

# or you can initialize a nifti object with a brain object or model object
bo_nii = se.load('example_data', return_type='nii')