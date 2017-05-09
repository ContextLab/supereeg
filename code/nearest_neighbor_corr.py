#!/usr/bin/env python

import numpy as np
import os
from stats import nearest_neighbors_corr
from bookkeeping import get_grand_parent_dir
import sys


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
                R_K_subj, nn_corr = nearest_neighbors_corr(fname, k_flat)
                np.savez(os.path.join(nn_corr_dir, file_name + '_nn_corr.npz'), nn_corr = nn_corr, R_K_subj = R_K_subj)
            else:
                print("only 1 electrode passes k = " + str(k_thresh))

        else:
            print("not enough electrodes pass k = " + str(k_thresh))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
