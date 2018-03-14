from builtins import str
import pytest
import os
import supereeg as se
import numpy as np
import pandas as pd
import nibabel as nib

bo = se.simulate_bo(n_samples=10, sample_rate=100)

nii = se.load('example_nifti')
bo_n = se.Brain(nii)

mo = se.load('example_model')
bo_m = se.Brain(mo)

def test_create_bo():
    assert isinstance(bo, se.Brain)

def test_bo_data_nifti():
    assert isinstance(bo_n, se.Brain)

def test_bo_data_model():
    assert isinstance(bo_m, se.Brain)

def test_bo_data_df():
    assert isinstance(bo.data, pd.DataFrame)

def test_bo_locs_df():
    assert isinstance(bo.locs, pd.DataFrame)

def test_bo_sessions_series():
    assert isinstance(bo.sessions, pd.Series)

def test_bo_nelecs_int():
    assert isinstance(bo.n_elecs, int)

def test_bo_nsecs_list():
    assert (bo.n_secs is None) or (type(bo.n_secs) is np.ndarray) or (type(bo.n_secs) is int) or (type(bo.n_secs) is float)

def test_bo_nsessions_int():
    assert isinstance(bo.n_sessions, int)

def test_bo_kurtosis_list():
    assert isinstance(bo.kurtosis, np.ndarray)

def test_samplerate_array():
    assert (bo.sample_rate is None) or (type(bo.sample_rate) is list)

def test_bo_getdata_nparray():
    assert isinstance(bo.get_data(), np.ndarray)

def test_bo_zscoredata_nparray():
    assert isinstance(bo.get_zscore_data(), np.ndarray)

def test_bo_get_locs_nparray():
    assert isinstance(bo.get_locs(), np.ndarray)

def test_bo_get_slice():
    bo_d = bo.get_slice(sample_inds=[1, 2], loc_inds=[1])
    assert isinstance(bo_d, se.Brain)
    assert bo_d.data.shape==(2,1)

def test_bo_resample():
    bo.resample(resample_rate=60)
    assert isinstance(bo, se.Brain)
    assert bo.sample_rate == [60]

def test_bo_save(tmpdir):
    p = tmpdir.mkdir("sub").join("example")
    print(p)
    print(type(p))
    bo.save(fname=p.strpath)
    test_bo = se.load(os.path.join(p.strpath + '.bo'))
    assert isinstance(test_bo, se.Brain)

def test_nii_nifti():
    assert isinstance(bo.to_nii(), se.Nifti)

def test_brain_load_str():
    bo = se.Brain('std')
    assert isinstance(bo, se.Brain)

def test_brain_brain():
    bo = se.simulate_bo(n_samples=10, sample_rate=100)
    bo = se.Brain(bo)
    assert isinstance(bo, se.Brain)

def test_brain_getitem():
    bo = se.simulate_bo(n_samples=10, sample_rate=100)
    bo = bo[:2]
    assert bo.data.shape[0]==2

def test_brain_getitem():
    bo = se.simulate_bo(n_samples=10, sample_rate=100)
    bos = [b for b in bo[:2]]
    assert all(isinstance(b, se.Brain) for b in bos)

## can't get tests for plots to work

# def test_bo_plot_locs(tmpdir):
#     p = tmpdir.mkdir("sub").join("example")
#     fig = bo.plot_locs(pdfpath=str(p))
#     assert os.path.exists(os.path.join(str(p), '.pdf'))
#     assert isinstance(fig, plt.Figure)
#
#
# def test_bo_plot_data(tmpdir):
#     p = tmpdir.mkdir("sub").join("example")
#     fig = bo.plot_data(filepath=str(p))
#     assert os.path.exists(os.path.join(str(p), '.png'))
#     assert isinstance(fig, plt.Figure)
