#!/usr/bin/env python

import numpy as np
import glob
import os
import sys
from bookkeeping import get_grand_parent_dir
import hypertools as hp
from scipy.spatial.distance import squareform as squareform
from stats import z2r
import re
from bookkeeping import get_grand_parent_dir, get_parent_dir
from nilearn import plotting as ni_plt

## input for this: directory for full correlation matrices, r and k_thresh
def main(r, k_thresh):
    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices_model')

    average_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'ave_model')

    cluster_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'cluster_model')
    if not os.path.isdir(cluster_dir):
        os.mkdir(cluster_dir)

    loc_name = 'R_small_MNI.npy'
    R_full = np.load(os.path.join(get_parent_dir(os.getcwd()), loc_name))

    if not os.path.isfile(os.path.join(cluster_dir, 'clustered_model_k_' + str(k_thresh) + '_r_' + str(r) + '.npz')):
        data = np.load(os.path.join(average_dir, 'average_model_k_' + str(k_thresh) + '_r_' + str(r) + '.npz'), mmap_mode='r')
        divide = np.divide(data['Numerator'], data['Denominator'])
        divide = squareform(z2r(divide).flatten(), checks=False)
        divide[np.where(np.isnan(divide))] = 0
        clustered = hp.tools.cluster(divide, 4)
        for c in np.unique(clustered):
            c_inds = [i for i, x in enumerate(clustered) if x == c]
            c_locs = R_full[c_inds]
            ni_plt.plot_connectome(np.eye(c_locs.shape[0]), c_locs, display_mode='lyrz', output_file= os.path.join(cluster_dir, 'R_cluster_' + str(c) + '.pdf'), node_kwargs={'alpha': 0.5, 'edgecolors': None}, node_size=10, node_color=np.ones(c_locs.shape[0]))

        ### save out r - once new matrix is added, need to convert back to z
        np.savez(outfile, sum_mat=sum_mat, n=count)
    else:
        print('average_full_matrix_k_' + str(k_thresh) + '_r_' + str(r) + '.npz', 'exists')

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])