#!/usr/bin/env python

import numpy as np
import glob
import os
from stats import rbf, good_chans, expand_corrmat, expand_matrix
from plot import compare_matrices
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir
from scipy.spatial.distance import squareform as squareform
import sys
from joblib import Parallel, delayed
import multiprocessing


## input: full path to file name, radius, kurtosis threshold

def main(fname, r, k_thresh):
    k_loc_name = 'R_full_k_' + str(k_thresh) + '_MNI.npy'

    ## create file name
    file_name = os.path.splitext(os.path.basename(fname))[0]
    ## existing directories:
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')
    fig_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'figs')

    ## check if cor_fig and full directories exist
    cor_fig_dir = os.path.join(fig_dir, 'sub_corrs')
    if not os.path.isdir(cor_fig_dir):
        os.mkdir(cor_fig_dir)

    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices')
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)

    # check if full location matrix (that pass specified threshold) exists
    if not os.path.isfile(os.path.join(full_dir, 'full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r) + '.npz')):
        if not os.path.isfile(os.path.join(get_parent_dir(os.getcwd()), k_loc_name)):
            k_full_locs = np.empty(shape=[0, 3])
            files = glob.glob(os.path.join(corr_dir, '*.npz'))  ## need to replace this - not sure if what should be in here
            for i in files:
                try:
                    sub_data = np.load(i, mmap_mode='r')
                    R_K_subj, C_K_subj = good_chans(sub_data['K_subj'], sub_data['R_subj'], k_thresh,
                                                    C = sub_data['C_subj'])
                    if R_K_subj.shape[0] > 1:
                        k_full_locs = np.append(k_full_locs, R_K_subj, axis=0)
                    else:
                        pass
                except:
                    pass
            unique_full_locs = np.vstack(set(map(tuple, k_full_locs)))
            np.save(os.path.join(get_parent_dir(os.getcwd()), k_loc_name), unique_full_locs[unique_full_locs[:, 0].argsort(),])

        ## load subject's electrodes
        sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'))
        R_subj = sub_data['R_subj']
        C_subj = sub_data['C_subj']
        K_subj = sub_data['K_subj']
        R_K_full = np.load(os.path.join(get_parent_dir(os.getcwd()), k_loc_name))

        # index R_subj and C_subj with K_subj
        R_K_subj, C_K_subj= good_chans(K_subj, R_subj, k_thresh, C = C_subj)

        ## check that atleast 2 electrodes pass kurtosis test
        if not R_K_subj == []:
            if R_K_subj.shape[0] > 1:
                ## create covariance matrices with kurtosis thresholed data
                # C_expand = expand_corrmat(R_full, R_K_subj, C_K_subj, float(r))
                ######### to use parallelize forloop:
                #### create weights matrix
                RBF_weights = rbf(R_K_full, R_K_subj, r)
                #### compile all pairs of coordidnates - loop over R_full matrix (lower triangle)
                inputs = [(x, y) for x in range(R_K_full.shape[0]) for y in range(x)]
                num_cores = multiprocessing.cpu_count()
                #### can't have the delayed(expand_corrmat) in a function
                results = Parallel(n_jobs=num_cores)(
                    delayed(expand_corrmat)(coord, R_K_subj, RBF_weights, C_K_subj) for coord in inputs)
                #### this expands the list from the mulitprocessor output  - the lower triangle of the matrix
                C_expand = expand_matrix(results, R_K_full)
                C_est = squareform(C_expand, checks=False)
                outfile = os.path.join(full_dir, 'full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r))
                np.savez(outfile + '.npz', C_est=C_est)
                sub_inds = get_rows(R_K_full, R_K_subj)
                C_est_sub = C_expand[sub_inds, :][:, sub_inds]
                outfile = os.path.join(cor_fig_dir, 'sub_cov_' + file_name + '_r_' + str(r) + '.png')
                C_K_subj[np.isnan(C_K_subj)] = 0
                compare_matrices(C_K_subj + np.eye(C_K_subj.shape[0]), C_est_sub, outfile, ('Observed' + file_name, 'Estimated' + file_name))
            else:
                print("not enough electrodes pass k = " + str(k_thresh))
    else:
        print('full_matrix_' + file_name + '_k' + str(k_thresh) + '_r' + str(r), 'exists')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])