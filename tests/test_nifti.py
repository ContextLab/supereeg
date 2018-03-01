import os
import supereeg as se
import numpy as np
import pandas as pd
import nibabel as nib

nii = se.load('example_nifti')

locs = np.array([[-61., -77.,  -3.],
                 [-41., -77., -23.],
                 [-21., -97.,  17.],
                 [-21., -37.,  77.],
                 [-21.,  63.,  -3.],
                 [ -1., -37.,  37.],
                 [ -1.,  23.,  17.],
                 [ 19., -57., -23.],
                 [ 19.,  23.,  -3.],
                 [ 39., -57.,  17.],
                 [ 39.,   3.,  37.],
                 [ 59., -17.,  17.]])

data = [se.simulate_model_bos(n_samples=10, sample_rate=10, locs=locs, sample_locs = 3) for x in range(2)]
# test model to compare
mo = se.Model(data=data, locs=locs)

bo = data[0]
nii_bo = se.Nifti(bo)
nii_mo = se.Nifti(mo)

def test_nifti():
    assert isinstance(nii, se.Nifti)
    assert issubclass(nii.__class__, nib.nifti1.Nifti1Image)


def test_nii_data_bo():
    assert isinstance(nii_bo, se.Nifti)

def test_nii_data_model():
    assert isinstance(nii_mo, se.Nifti)

def test_bo_save(tmpdir):
    p = tmpdir.mkdir("sub").join("example")
    print(p)
    print(type(p))
    nii.save(filepath=p.strpath)
    test_bo = se.load(os.path.join(p.strpath + '.nii'))
    assert isinstance(test_bo, se.Nifti)




# def test_nifti_to_bo():
#     bo = nii.to_bo()
#     assert isinstance(bo, se.Brain)
#
# def test_nifti_to_mo():
#     mo = nii.to_mo()
#     assert isinstance(mo, se.Model)
