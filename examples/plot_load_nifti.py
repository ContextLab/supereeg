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
import os

std_fname = os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/std.nii'
# load nifti -> brain object
nii = se.load('gray', vox_size=20)
#ni_plt.plot_anat(nii)
nifti = se.Nifti(std_fname)
ni_plt.plot_anat(nifti)

bo_b = nifti.to_bo()

bo_n = se.Brain(nii)
se_nii = se.Nifti(nii)


bo = se.Brain(se_nii)

# export brain object -> nifti
nifti = bo.to_nii(template='gray', vox_size=20)

# plot the result
ni_plt.plot_anat(nifti)
ni_plt.show()
