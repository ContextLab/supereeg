#!/usr/bin/env python

import numpy as np
import glob
import os
from stats import rbf, good_chans, expand_corrmat_j, expand_corrmat_parsed
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir, slice_list, partition_jobs
import sys
from scipy.spatial.distance import squareform as squareform
from plot import compare_matrices, plot_cov
from scipy.spatial.distance import cdist



## input: full path to file name, radius, kurtosis threshold, and number of matrix divisions

def main(fname, r, k_thresh):
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

    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices_model')
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)

    mod_fig_dir = os.path.join(fig_dir, 'model_matrices')
    if not os.path.isdir(mod_fig_dir):
        os.mkdir(mod_fig_dir)

    ## check if expanded subject level correlation matrix exists
    if not os.path.isfile(os.path.join(full_dir, file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_full_matrix' + '.npz')):

        ## load subject's electrodes
        sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'))
        R_subj = sub_data['R_subj'] # electrode locations
        C_subj = sub_data['C_subj'] # subject data
        K_subj = sub_data['K_subj'] # kurtosis - 1 by n_elecs
        R_full = np.load(os.path.join(get_parent_dir(os.getcwd()), loc_name))
        # R_K_full = np.load(os.path.join(get_parent_dir(os.getcwd()), k_loc_name))
        # index R_subj and C_subj with K_subj
        R_K_subj, C_K_subj, k_flat = good_chans(K_subj, R_subj, k_thresh, C = C_subj)

        ## check that atleast 2 electrodes pass kurtosis test
        if not R_K_subj == []:
            if R_K_subj.shape[0] > 1:
                RBF_weights = rbf(R_full, R_K_subj, r) # 3 by number of good channels
                C_K_subj[np.eye(C_K_subj.shape[0]) == 1] = 0
                K,W= expand_corrmat_j(RBF_weights, C_K_subj)
                C_expand = K/W
                outfile = os.path.join(mod_fig_dir, 'full_matrix_' + file_name + '_r_' + str(r) + '.png')
                plot_cov(C_expand, outfile=outfile)

                Numerator = squareform(K, checks=False)
                Denominator = squareform(W, checks=False)
                outfile = os.path.join(full_dir,
                                       file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_full_matrix')
                np.savez(outfile + '.npz', Numerator=Numerator, Denominator=Denominator)
                ### compare the expanded correlation matrix, but indexed by nearest locations - should match well
                nearest_inds = np.argmin(cdist(R_full, R_K_subj, metric='sqeuclidean'), axis=0)
                parsed_C = C_expand[nearest_inds, :][:, nearest_inds]
                parsed_C = parsed_C + np.eye(parsed_C.shape[0])
                outfile = os.path.join(mod_fig_dir, 'sub_matrix_compare_' + file_name + '_r_' + str(r) + '.png')
                compare_matrices(parsed_C, C_K_subj + np.eye(C_K_subj.shape[0]), outfile,
                                 ('Parsed_matrix' + file_name, 'Comparison' + file_name))

            else:
                print("not enough electrodes pass k = " + str(k_thresh))
    else:
        print(file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_full_matrix', 'exists')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])