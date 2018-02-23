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
from nilearn import plotting as ni_plt
from nilearn import image
from supereeg.helpers import _std, _gray
import nibabel as nib
import os
import hypertools as hyp


# load example data as brain object
bo = se.load('example_data')


# load example model
model = se.load('example_model')

bor = model.predict(bo)


# If no parameters are passed:
# default uses gray matter masked brain downsampled to 10 mm resolution

bor.plot_glass_brain(template='gray', vox_size=20)

