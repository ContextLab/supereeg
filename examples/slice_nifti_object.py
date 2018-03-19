# -*- coding: utf-8 -*-
"""
=============================
Slice nifti file
=============================

This example loads a brain object file, converts to a nifti, and slices first 3 timepoints.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se

# load example data
bo = se.load('example_data')

# check out the brain object (bo)
bo.info()

# convert bo to Nifti
bo_nii = se.Nifti(bo)

# slice nifti
nii_sliced = bo_nii.get_slice(index=[0,1,2])

nii_sliced.info()

