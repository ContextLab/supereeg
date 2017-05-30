#!/usr/bin/env python

import numpy as np
import glob
import os
from stats import rbf, good_chans, expand_corrmat, expand_matrix, r2z, z2r, compute_coord, expand_corrmat_j
from bookkeeping import get_rows, get_grand_parent_dir, get_parent_dir, slice_list, partition_jobs
import sys
from joblib import Parallel, delayed
import multiprocessing



## input: full path to file name, radius, kurtosis threshold, and number of matrix divisions

def main(fname, matrix_chunk, r, k_thresh, total_chunks):
    ## kurtosis pass union of electrode of locations
    #k_loc_name = 'R_full_k_' + str(k_thresh) + '_MNI.npy'
    ## downsampled locations with 5mm resolution:
    loc_name = 'R_full_MNI.npy'
    ## downsampled locations with 30mm resolution for sample data test:
    #loc_name = 'R_small_MNI.npy'

    ## create file name
    file_name = os.path.splitext(os.path.basename(fname))[0]
    ## existing directories:
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')
    fig_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'figs')

    ## check if cor_fig and full directories exist

    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices_chunked')
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)


    ## check if expanded subject level correlation matrix exists
    if not os.path.isfile(os.path.join(full_dir, file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_pooled_matrix_' + matrix_chunk.rjust(5,'0') + '.npy')):

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
                RBF_weights = rbf(R_full, R_K_subj, r) # 3 by number of good channels

                # sliced_up = slice_list([(x, y) for x in range(RBF_weights.shape[0]) for y in range(x)], int(total_chunks))[int(matrix_chunk)]
                #
                # Z = r2z(C_K_subj)
                # Z[np.isnan(Z)] = 0
                # results = Parallel(n_jobs=multiprocessing.cpu_count())(
                #     delayed(compute_coord)(coord, RBF_weights, Z) for coord in sliced_up)
                # outfile = os.path.join(full_dir, file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_pooled_matrix_' + matrix_chunk.rjust(5,'0'))
                # np.save(outfile, results)
                expand_corrmat_j(RBF_weights, C_K_subj)

            else:
                print("not enough electrodes pass k = " + str(k_thresh))
    else:
        print(file_name + '_k' + str(k_thresh) + '_r' + str(r)+ '_pooled_matrix_' + matrix_chunk.rjust(5,'0'), 'exists')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
