# -*- coding: utf-8 -*-
"""
=============================
Convert and save nifti file
=============================

This example converts a brain object into a nifti and saves it.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se


# load example data to convert to nifti
bo = se.load('example_data')

# convert to nifti two ways:

# if no parameters are passed default uses gray matter masked brain downsampled to 6 mm resolution
# 1: convert with to_nii() method
nii_bo1 = bo.to_nii(template='gray', vox_size=20)

nii_bo1.plot_glass_brain()

# 2: pass to initialize nifti
nii_bo2 = se.Nifti(bo, template='gray', vox_size=20)

nii_bo2.plot_glass_brain()

# save nifti
#nii_bo.save('/path/to/save/nifti')