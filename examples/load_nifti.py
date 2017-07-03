# -*- coding: utf-8 -*-
"""
=============================
Load a nifti file
=============================

This example loads a nifti file and converts it into a brain object.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import superEEG as se

nifti = se.load_nifti('/Users/andyheusser/Documents/github/superEEG/superEEG/data/MNI152_T1_6mm_brain.nii.gz')

print(nifti.info())
