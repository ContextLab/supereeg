#!/usr/bin/env python

import numpy as np
import glob
import os
import sys
from scipy.spatial.distance import squareform
from bookkeeping import get_grand_parent_dir
from stats import z2r, r2z
import re

## input for this: directory for full correlation matrices, r and k_thresh
def main(r, k_thresh):
	full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices')

	average_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'ave_matrices')
	if not os.path.isdir(average_dir):
		os.mkdir(average_dir)

	if not os.path.isfile(os.path.join(average_dir, 'average_full_matrix_k_' + str(k_thresh) + '_r_' + str(r) + '.npz')):
		files = glob.glob(os.path.join(full_dir, '*.npz'))
		print(files)
		results = []
		count = 0
		for i in files:
			matrix_variables = str('k' + str(k_thresh) + '_r' + str(r))
			match = re.search(matrix_variables, i)
			if match:
				count += 1
				data = np.load(i, mmap_mode='r')
				C_est = squareform(data['C_est'], checks=False)
				C_est[np.where(np.isnan(C_est))] = 0
				C_exp = r2z(C_est)
				next_mat = C_exp
				if np.shape(results)[0] == 0:
					results = next_mat
				else:
					results = results + next_mat
			else:
				pass
		average_matrix = z2r(results /count) + np.eye(np.shape(results)[0])
		outfile = os.path.join(average_dir, 'average_full_matrix_k_' + str(k_thresh) + '_r_' + str(r) + '.npz')
		np.savez(outfile, average_matrix=average_matrix, n=count)
	else:
		print('average_full_matrix_k_' + str(k_thresh) + '_r_' + str(r) + '.npz', 'exists')

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])