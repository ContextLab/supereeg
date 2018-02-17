import supereeg as se
import seaborn as sns
import numpy as np

import nilearn as nl

import os

from glob import glob
from scipy.io import loadmat
from skimage.transform import resize
import hypertools as hyp
from scipy.spatial.distance import pdist, cdist, squareform
from nilearn.plotting import plot_glass_brain, plot_anat
from matplotlib import pyplot as plt

std_brain = se.load('std')
std_nii = (se.load('std')).to_nii(template='std')

def resample_nii(x, target_res):
    def resample_3d(x, target_shape):        
        if len(target_shape) == 1:
            target_shape = target_shape * np.ones([1, 3])
        assert len(target_shape) == 3, 'resample_3d must resample to a 3-d matrix'

        x_reshaped = np.reshape(x, [np.prod(x.shape[0:3]), 1], order='F')
        #x_reshaped = np.arange(0, len(x_reshaped))[:, np.newaxis]
        y_reshaped = resize(x_reshaped, [np.prod(target_shape[0:3]), 1])

        return np.reshape(y_reshaped, (target_shape[2], target_shape[1], target_shape[0]), order='F')
    
    res = x.header.get_zooms()[0:3]
    shape = x.shape[0:3]
    
    scale = np.divide(res, target_res)    
    target_shape = (np.round(np.multiply(shape, scale))).astype(int)
        
    target_affine = x.affine
    target_affine[0:3, 0:3] *= scale
    target_affine[:, 3] *= np.append(scale, 1)
    
    return nb.nifti1.Nifti1Image(resample_3d(x.get_data(), target_shape), target_affine)

x = std_nii
target_res = 3
y = resample_nii(x, target_res)
plot_anat(y)

