#!/usr/bin/env python

import numpy as np
import glob
import os
from stats import rbf, good_chans, expand_corrmat_parsed, expand_matrix, r2z, z2r, compute_coord, expand_corrmat_j
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir, slice_list, partition_jobs
import sys
from joblib import Parallel, delayed
import multiprocessing



## input: full path to file name, radius, kurtosis threshold, and number of matrix divisions

def main(fname, r, k_thresh):

    loc_name = 'R_full_MNI.npy'

    ## create file name
    file_name = os.path.splitext(os.path.basename(fname))[0]
    ## existing directories:
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')
    average_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'ave_model')

    ## check if cor_fig and full directories exist

    check_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'check')
    if not os.path.isdir(check_dir):
        os.mkdir(check_dir)


    ## check if expanded subject level correlation matrix exists
    if not os.path.isfile(os.path.join(check_dir, file_name + '_k' + str(k_thresh) + '_r' + str(r) + '.npy')):

        ## load subject's electrodes
        sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'))
        R_subj = sub_data['R_subj'] # electrode locations
        C_subj = sub_data['C_subj'] # subject data
        K_subj = sub_data['K_subj'] # kurtosis - 1 by n_elecs
        R_full = np.load(os.path.join(get_parent_dir(os.getcwd()), loc_name))
        # R_K_full = np.load(os.path.join(get_parent_dir(os.getcwd()), k_loc_name))
        # index R_subj and C_subj with K_subj
        R_K_subj, C_K_subj= good_chans(K_subj, R_subj, k_thresh, C = C_subj)

        ## check that atleast 2 electrodes pass kurtosis test
        if not R_K_subj == []:
            if R_K_subj.shape[0] > 1:
                Full_locs = np.vstack((R_full, R_K_subj))
                RBF_weights = rbf(Full_locs, R_K_subj, r)
                expand_corrmat_j(RBF_weights, C_K_subj)
                ## load average matrix
                Ave_data = np.load(
                    os.path.join(average_dir, 'average_model' + '_k_' + str(k_thresh) + '_r_' + str(r) + '.npz'),
                    mmap_mode='r')

                #Ave_mat = squareform(Ave_data['matrix_sum'].flatten()/Ave_data['weights_sum'], checks= False)
                Ave_mat[np.eye(Ave_mat.shape[0]) == 1] = 0
                Ave_mat[np.where(np.isnan(Ave_mat))] = 0
                ## expand the altered average matrix to the new full set of locations (R_full + R_K_subj)
                RBF_weights = rbf(Full_locs, R_full, r) # 3 by number of good channels

                #### to test if the same expanding:
                K,W= expand_corrmat_j(RBF_weights, Ave_mat)
                Ave_expand = K/W
                Kp, Wp = expand_corrmat_parsed(RBF_weights, Ave_mat)
                Ave_expand_p = Kp / Wp


            else:
                print("not enough electrodes pass k = " + str(k_thresh))
    else:
        print(file_name + '_k' + str(k_thresh) + '_r' + str(r), 'exists')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
