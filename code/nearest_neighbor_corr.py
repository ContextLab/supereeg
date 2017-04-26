#!/usr/bin/env python

import numpy as np
import glob
import os
from stats import rbf, good_channels, processInput, expand_matrix, r2z, z2r, truncate, nearest_neighbors_corr
from plot import compare_matrices
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir
from scipy.spatial.distance import squareform as squareform
import sys
from joblib import Parallel, delayed
import multiprocessing
from sklearn.neighbors import NearestNeighbors

## input: full path to file name, kurtosis threshold
###### this returns a z transformed correlation value for each electrode that passes kurtosis test in this subject


def main(fname, k_thresh):

    ## create file name
    file_name = os.path.splitext(os.path.basename(fname))[0]

    nn_corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'nn_corr')
    if not os.path.isdir(nn_corr_dir):
        os.mkdir(nn_corr_dir)

    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')


    if not os.path.isfile(os.path.join(nn_corr_dir, file_name + '_nn_corr.npz')):
        ### check if it already exists, then check if the subject passes kurtosis thresholding
        sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'), mmap_mode='r')
        K_subj = sub_data['K_subj']

        k_flat = np.squeeze(np.where(K_subj < int(k_thresh)))

        if not k_flat == []:
            if np.shape(k_flat)[0] > 1:
                # pca = pca_describe_chunked(fname, k_flat)
                R_K_subj, nn_corr = nearest_neighbors_corr(fname, k_flat)
                np.savez(os.path.join(nn_corr_dir, file_name + '_nn_corr.npz'), nn_corr = nn_corr, R_K_subj = R_K_subj)
            else:
                print("only 1 electrode passes k = " + str(k_thresh))

        else:
            print("not enough electrodes pass k = " + str(k_thresh))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
