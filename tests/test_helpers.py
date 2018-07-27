from __future__ import print_function
from __future__ import division
from past.utils import old_div
import supereeg as se
import glob
from supereeg.helpers import *
from scipy.stats import kurtosis, zscore
import os

## don't understand why i have to do this:
from supereeg.helpers import _std, _gray, _resample_nii, _apply_by_file_index, _kurt_vals, _get_corrmat, _z2r, _r2z, \
    _log_rbf, \
    _timeseries_recon, _chunker, \
    _corr_column, _normalize_Y, _near_neighbor, _vox_size, _count_overlapping, _resample, \
    _nifti_to_brain, _brain_to_nifti, _to_log_complex, _to_exp_real, _logsubexp
from supereeg.model import _recover_model

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

# number of timeseries samples
n_samples = 10
# number of subjects
n_subs = 3
# number of electrodes
n_elecs = 5
# full brain object to parse and compare
bo_full = se.simulate_bo(n_samples=10, sessions=2, sample_rate=10, locs=locs)
# create brain object from subset of locations
sub_locs = bo_full.locs.iloc[6:]
sub_data = bo_full.data.iloc[:, sub_locs.index]
bo = se.Brain(data=sub_data.as_matrix(), sessions=bo_full.sessions, locs=sub_locs, sample_rate=10,
              meta={'brain object locs sampled': 2})
# simulate correlation matrix
data = [se.simulate_model_bos(n_samples=10, locs=locs, sample_locs=n_elecs) for x in range(n_subs)]
# test model to compare
test_model = se.Model(data=data, locs=locs, rbf_width=100)
bo_nii = se.Brain(_gray(20))
nii = _brain_to_nifti(bo_nii, _gray(20))

a = np.array([[1,2,3],[4,5,6],[7,8,9,]])
b = np.array([[-1,2,2],[-4,5,5],[-7,8,8,]])
c = np.array([[ 0,  4,  5], [ 0, 10, 11],[ 0, 16, 17]])
add_log = _to_log_complex(a)
a_log = _to_log_complex(a)
b_log = _to_log_complex(b)
c_log = _to_log_complex(c)
add_log.real = np.logaddexp(a_log.real,b_log.real)
add_log.imag = np.logaddexp(a_log.imag,b_log.imag)

def test_std():
    nii = _std(20)
    assert isinstance(nii, se.Nifti)

def test_gray():
    nii = _gray(20)
    assert isinstance(nii, se.Nifti)

def test_resample_nii():
    nii = _resample_nii(_gray(), 20, precision=5)
    assert isinstance(nii, se.Nifti)

def test_apply_by_file_index():
    def vstack_aggregate(prev, next):
        return np.max(np.vstack((prev, next)), axis=0)

    def kurtosis_xform(bo):
        return kurtosis(bo.data)

    max_kurtosis_vals = _apply_by_file_index(data[0], kurtosis_xform, vstack_aggregate)
    assert isinstance(max_kurtosis_vals, np.ndarray)

def test_kurt_vals():
    kurts_2 = _kurt_vals(data[0])
    assert isinstance(kurts_2, np.ndarray)

#NOTE: This test won't run because apply_by_file_index calls the kurtosis, but kurtosis doesnt support brain objects
# def test_kurt_vals_compare():
#     def aggregate(prev, next):
#         return np.max(np.vstack((prev, next)), axis=0)
#
#     kurts_1 = _apply_by_file_index(data[0], kurtosis, aggregate)
#     kurts_2 = _kurt_vals(data[0])
#     assert np.allclose(kurts_1, kurts_2)

def test_logsubexp():
    b_try = _to_exp_real(_logsubexp(c_log, a_log))
    assert np.allclose(b_try, b)

def test_get_corrmat():
    corrmat = _get_corrmat(data[0])
    assert isinstance(corrmat, np.ndarray)


def test_z_score():
    z_help = bo_full.get_zscore_data()
    z = np.vstack(
        (zscore(bo_full.get_data()[bo_full.sessions == 1]), zscore(bo_full.get_data()[bo_full.sessions == 2])))
    assert np.allclose(z, z_help)

def test_int_z2r():
    z = 1
    test_val = old_div((np.exp(2 * z) - 1), (np.exp(2 * z) + 1))
    input_val = _z2r(z)
    assert isinstance(input_val, (float, int))
    assert test_val == input_val

def test_array_z2r():
    z = np.array([1, 2, 3])
    test_val = old_div((np.exp(2 * z) - 1), (np.exp(2 * z) + 1))
    test_fun = _z2r(z)
    assert isinstance(test_fun, np.ndarray)
    assert np.allclose(test_val, test_fun)

def _r2z_z2r():
    z = np.array([1, 2, 3])
    test_fun = _r2z(_z2r(z))
    assert isinstance(test_fun, (int, np.ndarray))
    assert z == test_fun

def test_int_r2z():
    r = .1
    test_val = 0.5 * (np.log(1 + r) - np.log(1 - r))
    test_fun = _r2z(r)
    assert isinstance(test_fun, (float, int))
    assert test_val == test_fun

def test_array_r2z():
    r = np.array([.1, .2, .3])
    test_val = 0.5 * (np.log(1 + r) - np.log(1 - r))
    test_fun = _r2z(r)
    assert isinstance(test_fun, np.ndarray)
    assert np.allclose(test_val, test_fun)

def test_log_rbf():
    weights = _log_rbf(locs, locs[:10])
    assert isinstance(weights, np.ndarray)
    assert np.allclose(np.diag(weights), 0)

def test_tal2mni():
    tal_vals = tal2mni(locs)
    assert isinstance(tal_vals, np.ndarray)

def test_reconstruct():
    recon_test = test_model.predict(bo, nearest_neighbor=False, force_update=True)
    actual_test = bo_full.data.iloc[:, recon_test.locs.index]

    # actual_test: the true data
    # recon_test: the reconstructed data (using Model.predict)
    corr_vals = _corr_column(actual_test.as_matrix(), recon_test.data.as_matrix())
    assert np.all(corr_vals[~np.isnan(corr_vals)] <= 1) and np.all(corr_vals[~np.isnan(corr_vals)] >= -1)

def test_filter_elecs():
    bo_f = filter_elecs(bo)
    assert isinstance(bo_f, se.Brain)


def test_corr_column():
    X = np.matrix([[1, 2, 3], [1, 2, 3]])
    corr_vals = _corr_column(np.array([[.1, .4], [.2, .5], [.3, .6]]), np.array([[.1, .4], [.2, .5], [.3, .6]]))
    print(corr_vals)
    assert isinstance(corr_vals, (float, np.ndarray))

def test_normalize_Y():
    normed_y = _normalize_Y(np.array([[.1, .4], [.2, .5], [.3, .6]]))
    assert isinstance(normed_y, pd.DataFrame)
    assert normed_y.iloc[1][0] == 1.0
    assert normed_y.iloc[1][1] == 2.0

def test_model_compile(tmpdir):
    p = tmpdir.mkdir("sub")
    for m in range(len(data)):
        model = se.Model(data=data[m], locs=locs)
        model.save(fname=os.path.join(p.strpath, str(m)))
    model_data = glob.glob(os.path.join(p.strpath, '*.mo'))
    mo = se.Model(model_data)
    assert isinstance(mo, se.Model)
    assert np.allclose(mo.numerator.real, test_model.numerator.real, equal_nan=True)
    assert np.allclose(mo.numerator.imag, test_model.numerator.imag, equal_nan=True)
    assert np.allclose(mo.denominator, test_model.denominator, equal_nan=True)

def test_timeseries_recon():
    recon = _timeseries_recon(bo, test_model, 2)
    assert isinstance(recon, np.ndarray)
    assert np.shape(recon)[1] == np.shape(test_model.get_locs())[0]

def test_chunker():
    chunked = _chunker([1,2,3,4,5], 2)
    print(chunked)
    assert isinstance(chunked, list)
    assert chunked == [(1, 2), (3, 4), (5, None)]

def test_near_neighbor_auto():
    new_bo = _near_neighbor(bo, test_model, match_threshold='auto')
    assert isinstance(new_bo, se.Brain)

def test_near_neighbor_none():
    new_bo = _near_neighbor(bo, test_model, match_threshold=0)
    assert isinstance(new_bo, se.Brain)

def test_near_neighbor_int():
    new_bo = _near_neighbor(bo, test_model, match_threshold=10)
    assert isinstance(new_bo, se.Brain)

def test_vox_size():
    v_size = _vox_size(test_model.locs)
    assert isinstance(v_size, np.ndarray)

def test_count_overlapping():
    bool_overlap = _count_overlapping(bo_full.get_locs(), bo.get_locs())
    assert sum(bool_overlap)==bo.get_locs().shape[0]
    assert isinstance(bool_overlap, np.ndarray)

def test_resample():
    samp_data, samp_sess, samp_rate = _resample(bo, 8)
    assert isinstance(samp_data, pd.DataFrame)
    assert isinstance(samp_sess, pd.Series)
    assert isinstance(samp_rate, list)
    assert samp_rate==[8,8]

def test_nifti_to_brain():
    b_d, b_l, b_h = _nifti_to_brain(_gray(20))
    assert isinstance(b_d, np.ndarray)
    assert isinstance(b_l, np.ndarray)
    assert isinstance(b_h, dict)

def test_brain_to_nifti():
    nii = _brain_to_nifti(bo, _gray(20))
    assert isinstance(nii, se.Nifti)

def test_bo_nii_bo():
    nii = _brain_to_nifti(bo, _gray(20))
    b_d, b_l, b_h =_nifti_to_brain(nii)
    assert np.allclose(bo.get_locs(), b_l)

def test_nii_bo_nii():

    bo_nii = se.Brain(_gray(20))
    nii = _brain_to_nifti(bo_nii, _gray(20))
    nii_0 = _gray(20).get_data().flatten()
    nii_0[np.isnan(nii_0)] = 0
    assert np.allclose(nii_0, nii.get_data().flatten())
