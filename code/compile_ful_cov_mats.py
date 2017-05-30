#!/usr/bin/env python

import numpy as np
import glob
import os
from stats import rbf, good_chans, expand_corrmat, expand_matrix
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir, slice_list, get_nearest_rows
from scipy.spatial.distance import squareform as squareform
from plot import compare_matrices, plot_cov
from scipy.spatial.distance import cdist
import sys
import re




## input: full path to file name, radius, kurtosis threshold

def main(fname, r, k_thresh, total_chunks):
    ## kurtosis pass union of electrode of locations
    #k_loc_name = 'R_full_k_' + str(k_thresh) + '_MNI.npy'
    ## downsampled locations with 5mm resolution:
    #loc_name = 'R_full_MNI.npy'
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

    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices_chunked')
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)

    full_compiled_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices_compiled')
    if not os.path.isdir(full_compiled_dir):
        os.mkdir(full_compiled_dir)

    ## check if expanded subject level correlation matrix exists
    if not os.path.isfile(os.path.join(full_dir, 'full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r) + '.npz')):
        R_full = np.load(os.path.join(get_parent_dir(os.getcwd()), loc_name))
        ### to compare previous pooling method and new matrix division, this won't scale to cluster:
        # data_all = np.load(os.path.join(full_dir, file_name + '_k' + str(k_thresh) + '_r' + str(r) + '_all.npy'))
        # C_expand_old = expand_matrix(data_all, R_full)
        sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'))
        R_subj = sub_data['R_subj'] # electrode locations
        C_subj = sub_data['C_subj'] # subject data
        K_subj = sub_data['K_subj'] # kurtosis - 1 by n_elecs
        R_K_subj, C_K_subj = good_chans(K_subj, R_subj, k_thresh, C=C_subj)

        ###
        files = glob.glob(os.path.join(full_dir, '*.npy'))
        results = []
        count = 0
        chunk = []
        for i in files:
            matrix_variables = (file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_pooled_matrix_')
            match = re.search(matrix_variables, i)
            if match:
                count += 1
                if os.path.basename(i).count('_') == 5:
                    num_chunk = int(os.path.splitext(os.path.basename(i))[0].split("_", 5)[5])
                elif os.path.basename(i).count('_') == 6:
                    num_chunk = int(os.path.splitext(os.path.basename(i))[0].split("_", 6)[6])
                else:
                    return "error parsing chunk"
                data = np.load(i, mmap_mode='r')
                if np.shape(results)[0] == 0:
                    results = data
                    chunk = num_chunk
                else:
                    results = np.append(results,data)
                    chunk = np.append(chunk, num_chunk)
            else:
                pass
        if not int(total_chunks) == 1:
            expected_chunks = set(range(0, int(total_chunks)))
            actual_chunks = set(chunk)
            dif_chunk = actual_chunks.union(expected_chunks) - actual_chunks.intersection(expected_chunks)
            for d in dif_chunk:
                print('error: missing chunk file ' + file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_pooled_matrix_' + str(d).rjust(5,'0'))
        #### this expands the list from the mulitprocessor output  - the lower triangle of the matrix
        C_expand = expand_matrix(results, R_full)
        outfile = os.path.join(full_compiled_dir, 'full_matrix_results_' + file_name + '_k' + str(k_thresh) + '_r' + str(r))
        np.savez(outfile + '.npz', results = results, n = count)
        outfile = os.path.join(comp_fig_dir, 'full_matrix_' + file_name + '_r_' + str(r) + '.png')
        plot_cov(C_expand, outfile=outfile)
        C_est = squareform(C_expand, checks=False)
        outfile = os.path.join(full_compiled_dir, 'full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r))
        np.savez(outfile + '.npz', C_est=C_est)
        ### compare the expanded correlation matrix, but indexed by nearest locations - should match well
        nearest_inds = np.argmin(cdist(R_full, R_K_subj, metric='sqeuclidean'), axis=0)
        parsed_C = C_expand[nearest_inds,:][:, nearest_inds]
        outfile = os.path.join(comp_fig_dir, 'sub_matrix_compare_' + file_name + '_r_' + str(r) + '.png')
        compare_matrices(parsed_C, C_K_subj + np.eye(C_K_subj.shape[0]), outfile, ('Parsed_matrix' + file_name, 'Comparison' + file_name))
    else:
        print('full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r), 'exists')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])