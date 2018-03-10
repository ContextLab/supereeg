from builtins import str
from builtins import range
import supereeg as se
import numpy as np
import os
import nibabel as nib

bo = se.load('example_data')
bo_s = bo.get_slice(sample_inds=[0,1,2])

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

n_samples = 10
n_subs = 3
n_elecs = 10
data = [se.simulate_model_bos(n_samples=10, sample_rate=10, locs=locs,
                              sample_locs = n_elecs) for x in range(n_subs)]
test_bo = data[0]
test_model = se.Model(data=data, locs=locs)
bo = se.load('example_data')

def test_load_example_data():
    bo = se.load('example_data')
    assert isinstance(bo, se.Brain)

def test_load_example_filter():
    bo = se.load('example_filter')
    assert isinstance(bo, se.Brain)

def test_load_example_model():
    model = se.load('example_model')
    assert isinstance(model, se.Model)

def test_load_nifti():
    nii = se.load('example_nifti')
    assert isinstance(nii, nib.nifti1.Nifti1Image)

def test_load_pyFR_union():
    data = se.load('pyFR_union')
    assert isinstance(data, np.ndarray)

# def test_load_pyFR():
#     model = se.load('pyFR')
#     assert isinstance(model, se.Model)

def test_bo_load(tmpdir):
    p = tmpdir.mkdir("sub").join("example")
    test_bo.save(fname=p.strpath)
    bo = se.load(os.path.join(p.strpath + '.bo'))
    assert isinstance(bo, se.Brain)

def test_mo_load(tmpdir):
    p = tmpdir.mkdir("sub").join("example")
    test_model.save(fname=p.strpath)
    bo = se.load(os.path.join(p.strpath + '.mo'))
    assert isinstance(bo, se.Model)

def test_nii_load(tmpdir):
    p = tmpdir.mkdir("sub").join("example")
    test_bo.to_nii(filepath=p.strpath)
    nii = se.load(os.path.join(p.strpath + '.nii'))
    assert isinstance(nii, nib.nifti1.Nifti1Image)

def test_return_type_bo_with_bo():
    bo = se.load('example_data', return_type='bo')
    assert isinstance(bo, se.Brain)

def test_return_type_bo_with_mo():
    bo = se.load('example_model', return_type='bo')
    assert isinstance(bo, se.Brain)

def test_return_type_bo_with_nii():
    bo = se.load('example_nifti', return_type='bo')
    assert isinstance(bo, se.Brain)

def test_return_type_mo_with_bo():
    mo = se.load('example_data', return_type='mo')
    assert isinstance(mo, se.Model)

def test_return_type_mo_with_mo():
    mo = se.load('example_model', return_type='mo')
    assert isinstance(mo, se.Model)

# # passes nbut test takes a lot longer (like 10 seconds)
# def test_return_type_mo_with_nii():
#     mo = se.load('example_nifti', return_type='mo')
#     assert isinstance(mo, se.Model)

# passes
def test_return_type_nii_with_bo():
    nii = se.load('example_data', return_type='nii')
    assert isinstance(nii, se.Nifti)

# passes
def test_return_type_nii_with_mo():
    nii = se.load('example_model', return_type='nii')
    assert isinstance(nii, se.Nifti)

# passes
def test_return_type_nii_with_nii():
    nii = se.load('example_nifti', return_type='nii')
    assert isinstance(nii, se.Nifti)

def test_return_type_nii_with_bo_vs():
    nii = se.load(bo_s, vox_size = 20, return_type='nii')
    assert isinstance(nii, se.Nifti)

def test_return_type_nii_with_mo_vs():
    nii = se.load(bo_s, vox_size = 20, return_type='nii')
    assert isinstance(nii, se.Nifti)

def test_return_type_nii_with_nii_vs():
    nii = se.load(bo_s, vox_size = 20, return_type='nii')
    assert isinstance(nii, se.Nifti)