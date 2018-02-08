# -*- coding: utf-8 -*-
"""
=============================
Load nifti file
=============================

This example loads a nifti file and converts it into a brain object.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se
import os

# example nifti file
nifti_file = os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/gray_mask_20mm_brain.nii'

# load nifti as brain object
bo = se.load_nifti(nifti_file, mask_file=None)
