#!/usr/bin/env python


import numpy as np
from scipy.linalg import toeplitz as toeplitz
import pyDOE
from bookkeeping import get_rows


def synthesize(dims, fname, res=10):
    n = np.prod(dims)

    #ground truth covariance matrix and locations
    K_full = toeplitz(np.arange(n, 0, -1)) / float(n)
    R_full = np.array(pyDOE.fullfact(dims))

    #subsample R_full to get subject data
    R_subj = R_full[0::res, :]

    inds = get_rows(R_full, R_subj)
    K_subj = K_full[inds, :][:, inds]

    np.savez(fname, K_full=K_full, K_subj=K_subj, R_full=R_full, R_subj=R_subj)
