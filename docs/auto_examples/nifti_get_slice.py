# -*- coding: utf-8 -*-
"""
=============================
Index nifti object
=============================

In this example, we load a brain object as a nifti object, and index 5 timepoints.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# load
import supereeg as se

#  initialize a nifti object with a brain object or model object
bo_nii = se.load('example_data', return_type='nii')

# or you can slice first 5 time points
bo_nii_slice = bo_nii.get_slice(index=[0,1,2,3,4])


