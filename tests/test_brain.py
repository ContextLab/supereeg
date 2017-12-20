# -*- coding: utf-8 -*-

import pytest
import superEEG as se
import numpy as np
import pandas as pd
import nibabel as nib

data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=100)
locs = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=10)
bo = se.Brain(data=data, locs=locs)

def test_create_bo():
    assert isinstance(bo, se.Brain)

def test_bo_data_df():
    assert isinstance(bo.data, pd.DataFrame)


def test_bo_locs_df():
    assert isinstance(bo.locs, pd.DataFrame)


def test_bo_sessions_series():
    assert isinstance(bo.sessions, pd.Series)

def test_bo_nelecs_int():
    assert isinstance(bo.n_elecs, int)

def test_bo_nsecs_list():
    assert (bo.n_secs is None) or (type(bo.n_secs) is np.ndarray)

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

def test_nii_nifti():
    assert isinstance(bo.to_nii(), nib.nifti1.Nifti1Image)