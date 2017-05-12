#!/usr/bin/env python

import numpy as np
import glob
import os
from stats import tal2mni, corrmat, kurt_vals
from bookkeeping import get_grand_parent_dir, get_parent_dir
import sys


def main(fname):
    ## downsampled locations with 5mm resolution:
    # loc_name = 'R_full_MNI.npy'
    ## downsampled locations with 30mm resolution for sample data test:
    loc_name = 'R_small_MNI.npy'

    ## create file name
    file_name = os.path.splitext(os.path.basename(fname))[0]

    ## check if corr, fig, and full directories exist
    corr_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'corr_matrices')
    if not os.path.isdir(corr_dir):
        os.mkdir(corr_dir)

    fig_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'figs')
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    cor_fig_dir = os.path.join(fig_dir, 'sub_corrs')
    if not os.path.isdir(cor_fig_dir):
        os.mkdir(cor_fig_dir)

    # check if subject level correlation matrix exists
    if not os.path.isfile(os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz')):
        #### we are using a different set of locations, downsampled from a standard brain - this would be to create it from the union of all electrodes
        # if not os.path.isfile(os.path.join(get_parent_dir(os.getcwd()), loc_name)):
        #     full_locs = np.empty(shape=[0, 3])
        #     files = glob.glob(os.path.join(os.path.dirname(fname),'*.npz'))  ## need to replace this - not sure if what should be in here
        #     for i in files:
        #         try:
        #             data = np.load(i, mmap_mode='r')
        #             tempR = tal2mni(data['R'])
        #             full_locs = np.append(full_locs, tempR, axis=0)
        #         except:
        #             pass
        #     unique_full_locs = np.vstack(set(map(tuple, full_locs)))
        #     np.save(os.path.join(get_parent_dir(os.getcwd()), loc_name), unique_full_locs[unique_full_locs[:, 0].argsort(),])

        ## create subject correlation matrix
        C_subj = corrmat(fname)
        ## calculate max kurtosis for each channel
        K_subj = kurt_vals(fname)
        outfile = os.path.join(corr_dir, 'sub_corr_' + file_name + '.npz')
        ## save correlation matrix and kurtosis values
        data = np.load(fname, mmap_mode='r')
        R_subj = tal2mni(data['R'])
        np.savez(outfile, C_subj=C_subj, K_subj=K_subj, R_subj=R_subj)

    else:
        print('sub_corr_' + file_name, 'exists')



if __name__ == "__main__":
    main(sys.argv[1])