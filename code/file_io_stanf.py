import numpy as np
import scipy.io as io
import os

def loadmat(fname):
	data = io.loadmat(fname)
	k = data.keys()
	if k == []:
		print(fname, 'Error loading scipy')
	else:
		data = {'Y': data['data'], 'R': data['locs']}
	return data

def mat2npz(infile, outfile):
	data = loadmat(infile)
	file_name = os.path.splitext(os.path.basename(infile))[0]
	np.savez(outfile + file_name + '.npz', Y = data['Y'], R = data['R'])
