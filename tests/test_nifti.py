import os
import supereeg as se
import numpy as np
import pandas as pd
import nibabel as nib

nii = se.load('example_nifti')

mo = nii.to_mo()

def test_nifti():
    assert isinstance(nii, se.Nifti)
    assert issubclass(nii.__class__, nib.nifti1.Nifti1Image)

def test_nifti_to_bo():
    bo = nii.to_bo()
    assert isinstance(bo, se.Brain)

def test_nifti_to_mo():
    mo = nii.to_mo()
    assert isinstance(mo, se.Model)
