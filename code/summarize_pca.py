#!/usr/bin/env python

import os
import sys
import numpy as np
from bookkeeping import  get_grand_parent_dir
from stats import pca_describe_var


def main(fname, k_thresh):
    file_name = os.path.splitext(os.path.basename(fname))[0]
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')

    pca_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'pca')
    if not os.path.isdir(pca_dir):
        os.mkdir(pca_dir)

    if not os.path.isfile(os.path.join(pca_dir, file_name + '_pca.npz')):
        ### check if it already exists, then check if the subject passes kurtosis thresholding
        sub_data = np.load(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz'), mmap_mode='r')
        K_subj = sub_data['K_subj']

        k_flat = np.squeeze(np.where(K_subj < int(k_thresh)))

        if not k_flat == []:
            if np.shape(k_flat)[0] > 1:
                pca = pca_describe_var(fname, k_flat)
                np.savez(os.path.join(pca_dir, file_name + '_pca.npz'), PCA = pca)
            else:
                print("only 1 electrode passes k = " + str(k_thresh))

        else:
            print("not enough electrodes pass k = " + str(k_thresh))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
