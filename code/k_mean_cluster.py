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

## input for this: directory for full correlation matrices, r and k_thresh
def main(r, k_thresh):
    full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices_model')

    average_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'ave_model')

    cluster_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'cluster_model')
    if not os.path.isdir(cluster_dir):
        os.mkdir(cluster_dir)

    if not os.path.isfile(os.path.join(cluster_dir, 'clustered_model_k_' + str(k_thresh) + '_r_' + str(r) + '.npz')):
        data = np.load(os.path.join(average_dir, 'average_model_k_' + str(k_thresh) + '_r_' + str(r) + '.npz'), mmap_mode='r')
        divide = np.divide(data['Numerator'], data['Denominator'])
        clustered = hp.tools.cluster(divide, 4)
        #average_matrix = z2r(results /weights) + np.eye(np.shape(results)[0])
        outfile = os.path.join(average_dir, 'average_model_k_' + str(k_thresh) + '_r_' + str(r) + '.npz')
        ### save out r - once new matrix is added, need to convert back to z
        np.savez(outfile, sum_mat=sum_mat, n=count)
    else:
        print('average_full_matrix_k_' + str(k_thresh) + '_r_' + str(r) + '.npz', 'exists')

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])