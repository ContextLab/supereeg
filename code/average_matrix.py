#!/usr/bin/env python

import numpy as np
import glob
import os
import sys
from bookkeeping import get_grand_parent_dir
import re

## input for this: directory for full correlation matrices, r and k_thresh
def main(r, k_thresh):
	full_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'full_matrices_model')

	average_dir = os.path.join(get_grand_parent_dir(os.getcwd()), 'ave_model')
	if not os.path.isdir(average_dir):
		os.mkdir(average_dir)

	if not os.path.isfile(os.path.join(average_dir, 'average_model_k_' + str(k_thresh) + '_r_' + str(r) + '.npz')):
		files = glob.glob(os.path.join(full_dir, '*.npz'))
		print(files)
		numerator = []
		denominator = []
		count = 0
		for i in files:
			matrix_variables = str('k' + str(k_thresh) + '_r' + str(r))
			match = re.search(matrix_variables, i)
			if match:
				count += 1
				data = np.load(i, mmap_mode='r')
				#C_est = squareform(data['C_est'], checks=False)
				next_num = data['Numerator']
				next_denom = data['Denominator']
				if np.shape(numerator)[0] == 0:
					numerator = next_num
					denominator = next_denom
				else:
					numerator = np.nansum(np.dstack((numerator, next_num)),2)
					denominator += next_denom
			else:
				pass
		#average_matrix = z2r(results /weights) + np.eye(np.shape(results)[0])
		outfile = os.path.join(average_dir, 'average_model_k_' + str(k_thresh) + '_r_' + str(r) + '.npz')
		### save out r - once new matrix is added, need to convert back to z
		np.savez(outfile, Numerator=numerator, Denominator=denominator, n=count)
	else:
		print('average_full_matrix_k_' + str(k_thresh) + '_r_' + str(r) + '.npz', 'exists')

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])