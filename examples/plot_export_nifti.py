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


# load example nifti
nii = se.load('example_nifti')

# or load example data to convert to nifti
bo = se.load('example_data')

# convert to nifti two ways:
# If no parameters are passed:
# default uses gray matter masked brain downsampled to 6 mm resolution

# 1: convert with brain object method
nii_bo1 = bo.to_nii(template='gray', vox_size=20)

# 2: pass to initialize nifti
nii_bo2 = se.Nifti(bo, template='gray', vox_size=20)

# save nifti
#nii_bo2.save('/path/to/save/nifti')