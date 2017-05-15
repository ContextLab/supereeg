#!/usr/bin/env python

import numpy as np
import glob
import os
from stats import rbf, good_chans, expand_corrmat, expand_matrix
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir, slice_list
from scipy.spatial.distance import squareform as squareform
from plot import compare_matrices
import sys
import re


## input: full path to file name, radius, kurtosis threshold

def main(fname, r, k_thresh):
    ## kurtosis pass union of electrode of locations
    #k_loc_name = 'R_full_k_' + str(k_thresh) + '_MNI.npy'
    ## downsampled locations with 5mm resolution:
    # loc_name = 'R_full_MNI.npy'
    ## downsampled locations with 30mm resolution for sample data test:
    loc_name = 'R_small_MNI.npy'

    ## create file name
    file_name = os.path.splitext(os.path.basename(fname))[0]
    ## existing directories:
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')
    fig_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'figs')

    ## check if cor_fig and full directories exist
    comp_fig_dir = os.path.join(fig_dir, 'compare_matrices')
    if not os.path.isdir(comp_fig_dir):
        os.mkdir(comp_fig_dir)

    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices')
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)

    ## check if expanded subject level correlation matrix exists
    if not os.path.isfile(os.path.join(full_dir, 'full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r) + '.npz')):
        R_full = np.load(os.path.join(get_parent_dir(os.getcwd()), loc_name))

        ### to compare previous pooling method and new matrix division, this won't scale to cluster:
        data_all = np.load(os.path.join(full_dir, file_name + '_k' + str(k_thresh) + '_r' + str(
                    r) + '_all.npy'))
        C_expand_old = expand_matrix(data_all, R_full)

        ###
        files = glob.glob(os.path.join(full_dir, '*.npy'))
        results = []
        count = 0
        for i in files:
            matrix_variables = (file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_pooled_matrix_')
            match = re.search(matrix_variables, i)
            if match:
                count += 1
                data = np.load(i, mmap_mode='r')
                if np.shape(results)[0] == 0:
                    results = data
                else:
                    results = np.append(results,data)
            else:
                pass

        #### this expands the list from the mulitprocessor output  - the lower triangle of the matrix
        C_expand = expand_matrix(results, R_full)
        C_est = squareform(C_expand, checks=False)
        outfile = os.path.join(full_dir, 'full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r))
        np.savez(outfile + '.npz', C_est=C_est)
        outfile = os.path.join(comp_fig_dir, 'sub_matrix_compare_' + file_name + '_r_' + str(r) + '.png')
        compare_matrices(C_expand, C_expand_old, outfile, ('Parsed_matrix' + file_name, 'Comparison' + file_name))
    else:
        print('full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r), 'exists')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])