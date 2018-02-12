from __future__ import print_function
from __future__ import division
from past.utils import old_div
from builtins import bytes
import pytest
import supereeg as se
import glob
from supereeg.helpers import *
from scipy.stats import kurtosis, zscore
import os

## don't understand why i have to do this:
from supereeg.helpers import _apply_by_file_index, _kurt_vals, _get_corrmat, _z2r, _r2z, _rbf, _uniquerows, \
    _expand_corrmat_fit, _expand_corrmat_predict, _chunk_bo, _timeseries_recon, _chunker, _reconstruct_activity, \
    _round_it, _corr_column, _normalize_Y, _near_neighbor, _vox_size

locs = se.load('example_locations')[0::17]
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
test_model = se.Model(data=data, locs=locs)

def test__apply_by_file_index():
    def aggregate(prev, next):
        return np.max(np.vstack((prev, next)), axis=0)

    kurts_1 = _apply_by_file_index(data[0], kurtosis, aggregate)
    assert isinstance(kurts_1, np.ndarray)

def test__kurt_vals():
    kurts_2 = _kurt_vals(data[0])
    assert isinstance(kurts_2, np.ndarray)

def test__kurt_vals_compare():
    def aggregate(prev, next):
        return np.max(np.vstack((prev, next)), axis=0)

    kurts_1 = _apply_by_file_index(data[0], kurtosis, aggregate)
    kurts_2 = _kurt_vals(data[0])
    assert np.allclose(kurts_1, kurts_2)

def test_get_corrmat():
    corrmat = _get_corrmat(data[0])
    assert isinstance(corrmat, np.ndarray)

def test_int_z2r():
    z = 1
    test_val = old_div((np.exp(2 * z) - 1), (np.exp(2 * z) + 1))
    input_val = _z2r(z)
    assert isinstance(input_val, (float, int))
    assert test_val == input_val

def test_array_z2r():
    z = [1, 2, 3]
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

def test_rbf():
    weights = _rbf(locs, locs[:10])
    weights_same = _rbf(locs[:10], locs[:10], 1)
    assert isinstance(weights, np.ndarray)
    assert np.allclose(weights_same, np.eye(np.shape(weights_same)[0]))

def test_tal2mni():
    tal_vals = tal2mni(locs)
    assert isinstance(tal_vals, np.ndarray)

def test_uniquerows():
    full_locs = np.concatenate((locs, locs[:10]), axis=0)
    test_fun = _uniquerows(full_locs)
    assert isinstance(test_fun, np.ndarray)
    assert np.shape(test_fun) == np.shape(locs)

def test_expand_corrmat_fit():
    sub_corrmat = _get_corrmat(bo)
    np.fill_diagonal(sub_corrmat, 0)
    sub_corrmat = _r2z(sub_corrmat)
    weights = _rbf(test_model.locs, bo.locs)
    expanded_num_f, expanded_denom_f = _expand_corrmat_fit(sub_corrmat, weights)

    assert isinstance(expanded_num_f, np.ndarray)
    assert isinstance(expanded_denom_f, np.ndarray)
    assert np.shape(expanded_num_f)[0] == test_model.locs.shape[0]


def test_expand_corrmat_predict():
    sub_corrmat = _get_corrmat(bo)
    np.fill_diagonal(sub_corrmat, 0)
    sub_corrmat = _r2z(sub_corrmat)
    weights = _rbf(test_model.locs, bo.locs)
    expanded_num_p, expanded_denom_p = _expand_corrmat_predict(sub_corrmat, weights)

    assert isinstance(expanded_num_p, np.ndarray)
    assert isinstance(expanded_denom_p, np.ndarray)
    assert np.shape(expanded_num_p)[0] == test_model.locs.shape[0]

def test_expand_corrmats_same():
    sub_corrmat = _get_corrmat(bo)
    np.fill_diagonal(sub_corrmat, 0)  # <- possible failpoint
    sub_corrmat_z = _r2z(sub_corrmat)
    weights = _rbf(test_model.locs, bo.locs)

    expanded_num_p, expanded_denom_p = _expand_corrmat_predict(sub_corrmat_z, weights)
    model_corrmat_p = np.divide(expanded_num_p, expanded_denom_p)
    expanded_num_f, expanded_denom_f = _expand_corrmat_predict(sub_corrmat_z, weights)
    model_corrmat_f = np.divide(expanded_num_f, expanded_denom_f)

    np.fill_diagonal(model_corrmat_f, 0)
    np.fill_diagonal(model_corrmat_p, 0)

    s = test_model.locs.shape[0] - bo.locs.shape[0]

    Kba_p = model_corrmat_p[:s, s:]
    Kba_f = model_corrmat_f[:s, s:]
    Kaa_p = model_corrmat_p[s:, s:]
    Kaa_f = model_corrmat_f[s:, s:]

    assert isinstance(Kaa_p, np.ndarray)
    assert isinstance(Kaa_f, np.ndarray)
    assert np.allclose(Kaa_p, Kaa_f)
    assert np.allclose(Kba_p, Kba_f)

def test_reconstruct():
    recon_test = test_model.predict(bo, nearest_neighbor=False, force_update=True)
    actual_test = bo_full.data.iloc[:, recon_test.locs.index]
    zbo = copy.copy(bo)
    zbo.data = pd.DataFrame(bo.get_zscore_data())
    mo = test_model.update(zbo)
    model_corrmat_x = np.divide(mo.numerator, mo.denominator)
    model_corrmat_x = _z2r(model_corrmat_x)
    np.fill_diagonal(model_corrmat_x, 0)
    #recon_data = np.hstack((_timeseries_recon(zbo, model_corrmat_x), zscore(bo.data.as_matrix())))
    recon_data = _timeseries_recon(zbo, model_corrmat_x)
    corr_vals = _corr_column(actual_test.as_matrix(), recon_test.data.as_matrix())
    assert isinstance(recon_data, np.ndarray)
    assert np.allclose(recon_data, recon_test.data)
    assert 1 >= corr_vals.mean() >= -1

def test_round_it():
    rounded_array = _round_it(np.array([1.0001, 1.99999]), 3)
    rounded_float = _round_it(1.0009, 3)
    assert isinstance(rounded_array, (int, float, np.ndarray))
    assert np.allclose(rounded_array, np.array([1, 2]))
    assert rounded_float == 1.001

def test_filter_elecs():
    bo_f = filter_elecs(bo)
    assert isinstance(bo_f, se.Brain)

def test_filter_subj():
    bo_s = filter_subj(bo)
    bo_f = filter_subj(bo_full)
    assert isinstance(bo_s, (str, dict, type(None)))
    assert isinstance(bo_f, (str, dict, type(None)))

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

### not sure how to write tests for the Nifti conversion functions

def test_model_compile(tmpdir):
    p = tmpdir.mkdir("sub")
    for m in range(len(data)):
        model = se.Model(data=data[m], locs=locs)
        model.save(fname=os.path.join(p.strpath, str(m)))

    model_data = glob.glob(os.path.join(p.strpath, '*.mo'))
    mo = se.model_compile(model_data)
    assert isinstance(mo, se.Model)
    assert np.allclose(mo.numerator, test_model.numerator)
    assert np.allclose(mo.denominator, test_model.denominator)

def test_chunk_bo():
    chunk = tuple([1,2,3])
    chunked_bo = _chunk_bo(bo_full, chunk)
    print(type(_chunk_bo))
    assert isinstance(chunked_bo, se.Brain)
    assert np.shape(chunked_bo.data)[0]==np.shape(chunk)[0]

def test_timeseries_recon():
    mo = np.divide(test_model.numerator, test_model.denominator)
    np.fill_diagonal(mo, 0)
    recon = _timeseries_recon(bo, mo, 2)
    assert isinstance(recon, np.ndarray)
    assert np.shape(recon)[1] == np.shape(mo)[1]

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

def test_sort_unique_locs():
    sorted = sort_unique_locs(locs)
    assert isinstance(sorted, np.ndarray)
