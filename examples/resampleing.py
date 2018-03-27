# -*- coding: utf-8 -*-
"""
=============================
Resampling
=============================

This example shows you how to resample your data

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se

# load example data
bo = se.load('example_data')

# info contains sample rate
bo.info()

# default resample returns the brain object
bo.resample()

# show new info - nothing changed if resample_rate isn't specified
bo.info()

# resample to specified sample rate
bo.resample(100)

# show new info
bo.info()

# can also change sample rate when converting to nifti image
nii = bo.to_nii(template='gray', vox_size=20, sample_rate=64)







