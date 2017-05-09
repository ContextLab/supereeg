import numpy as np
import h5py as h5
import scipy.io as io
import os

def loadmat(fname):
	def h5py_loadmat(fname):
		data = h5.File(fname, 'r')
		k = data.keys()
		if k == []:
			print(fname, 'Error loading H5')
		else:
			data = {'Y': np.array(data['Y']).T, 'R': np.array(data['R']).T, 'fname_labels': np.array(data['fname_labels']).T, 'samplerate': np.array(data['samplerate']).T}
		return data

	def scipy_loadmat(fname):
		data = io.loadmat(fname)
		k = data.keys()
		if k == []:
			print(fname, 'Error loading scipy')
		else:
			data = {'Y': data['Y'], 'R': data['R'], 'fname_labels': data['fname_labels'], 'samplerate': data['samplerate']}
		return data

	try:
		data = h5py_loadmat(fname)
	except:
		try:
			data = scipy_loadmat(fname)
		except:
			raise Exception('UNKNOWN FORMAT: ' + fname)
	return data

def mat2npz(infile, outfile):
	data = loadmat(infile)
	file_name = os.path.splitext(os.path.basename(infile))[0]
	np.savez(outfile + file_name + '.npz', Y = data['Y'], R = data['R'], fname_labels = data['fname_labels'], samplerate = data['samplerate'])


