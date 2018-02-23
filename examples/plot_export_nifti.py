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

nifti_a = bo.to_nii(vox_size=20)
a = image.index_img(nifti_a, 1)
ni_plt.plot_glass_brain(a)

# If no template parameter is passed, uses gray matter masked

# You can specify integer or float as voxel size for nifti export
nifti_b = bo.to_nii(vox_size=6)
b = image.index_img(nifti_b, 1)
ni_plt.plot_glass_brain(b)


# Or you can specify a list as voxel size
nifti_c = bo.to_nii(vox_size=[10, 10, 10])
c = image.index_img(nifti_c, 1)
ni_plt.plot_glass_brain(c)

# You can also specify a template with specific voxel sizes
nifti_d = bo.to_nii(template=_gray([10, 10, 10]))
d = image.index_img(nifti_d, 1)
ni_plt.plot_glass_brain(d)
