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


# load Nifti
# example nifti is the gray matter masked MNI152 brain downsampled to 20mm
nii = se.load('example_nifti')

# plot nifti
nii.plot_anat()

# create Nifti with affine and dataobj
af = nii.affine
do = nii.dataobj
make_nii= se.Nifti(do, affine=af)


# nifti -> brain object - initialize brain object with nifti object
bo = se.Brain(nii)

# plot brain object
bo.plot_data()

# export brain object -> nifti
nii_r = bo.to_nii(template='gray', vox_size=20)

# plot the result (same as before)
nii_r.plot_anat()

# or initialize nifti object with brain object
nii_bo = se.Nifti(bo)

# # plot the result (same as before)
nii_bo.plot_anat()