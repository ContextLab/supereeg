import os
import sys
import pickle
import scipy
import numpy as np
import nibabel as nb
import deepdish as dd
import pandas as pd
from nilearn.input_data import NiftiMasker
from scipy.spatial.distance import squareform
from .brain import Brain
from .model import Model
from ._helpers.stats import tal2mni
from ._helpers.stats import z2r
from ._helpers.stats import r2z

def load(fname):
    """
    Load nifti file, brain or model object, or example data.

    This function can load in example data, as well as nifti files, brain objects
    and model objects by detecting the extension and calling the appropriate
    load function.  Thus, be sure to include the file extension in the fname
    parameter.

    Parameters
    ----------
    fname : string
        The name of the example data or a filepath.  Example data includes:
        example_data, example_model and example_locations

    Returns
    ----------
    data : nibabel.Nifti1, superEEG.Brain or superEEG.Model
        Data to be returned

    """
    # if sys.version_info[0]==3:
    #     pickle_options = {
    #         'encoding' : 'latin1'
    #     }
    # else:
    #     pickle_options = {}
    # if dataset is 'example_data':
    #     fileid = '0B7Ycm4aSYdPPREJrZ2stdHBFdjg'
    #     url = 'https://docs.google.com/uc?export=download&id=' + fileid
    #     data = pickle.loads(requests.get(url, stream=True).content, **pickle_options)

    # load example data
    if fname is 'example_data':
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/CH003.npz', 'rb') as handle:
            f = np.load(handle)
            data = f['Y']
            sample_rate = f['samplerate']
            sessions = f['fname_labels']
            locs = tal2mni(f['R'])
            meta = 'CH003'

        return Brain(data=data, locs=locs, sessions=sessions, sample_rate=sample_rate, meta= meta)

    # load example model
    ## the example model needs to be replaced with the one on the cluster
    elif fname is 'example_model':
        try:
            with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/mini_model.mo', 'rb') as handle:
                example_model = pickle.load(handle)
            return example_model
        except:
            model = pd.read_pickle(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/mini_model.mo')
            return model

    elif fname is 'pyFR_k10r20_20mm':
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/example_model_k_10_r_20.npz', 'rb') as handle:
            f = np.load(handle)
            numerator = squareform(f['Numerator'].flatten())
            denominator = squareform(f['Denominator'].flatten())
            n_subs = f['n']

        with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/gray_20mm_locs.npy', 'rb') as handle:
            l = np.load(handle)

        return Model(numerator=numerator, denominator=denominator, n_subs=n_subs, locs=pd.DataFrame(l, columns=['x', 'y', 'z']))

    ## This should be replaced with the new 10K model and named pyFR_k10r20_6mm

    # elif fname is 'pyFR_k10r20_8mm':
    #     with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/pyFR_k10r20.npz', 'rb') as handle:
    #         f = np.load(handle)
    #         numerator = squareform(f['Numerator'].flatten())
    #         denominator = squareform(f['Denominator'].flatten())
    #         n_subs = f['n']
    #
    #     with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/gray_8mm_locs.npy', 'rb') as handle:
    #         l = np.load(handle)
    #
    #     return Model(numerator=numerator, denominator=denominator, n_subs=n_subs, locs=pd.DataFrame(l, columns=['x', 'y', 'z']))


    # load example locations
    elif fname is 'example_locations':
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/gray_20mm_locs.npy', 'rb') as handle:
            locs = np.load(handle)
        return locs

    elif fname is 'example_nifti':
        bo = load_nifti(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/gray_mask_8mm_brain.nii')
        return bo

    elif fname is 'pyFR_union':
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/pyFR_k10_locs.npz', 'rb') as handle:
            data = np.load(handle)
            locs = data['locs']
            print('subjects = ', data['subjs'])
        return locs

    elif fname is 'mini_model':
        bo = load_nifti(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/gray_mask_20mm_brain.nii')
        return bo

    elif fname is 'gray_mask_6mm_brain':
        bo = load_nifti(os.path.dirname(os.path.abspath(__file__)) + '/../superEEG/data/gray_mask_6mm_brain.nii')
        return bo

    # load brain object
    elif fname.split('.')[-1]=='bo':
        bo = dd.io.load(fname)
        return Brain(data=bo['data'], locs=bo['locs'], sessions=bo['sessions'],
                     sample_rate=bo['sample_rate'], meta=bo['meta'],
                     date_created=bo['date_created'])

    # load model object
    elif fname.split('.')[-1]=='mo':
        mo = dd.io.load(fname)
        return Model(numerator=mo['numerator'], denominator=mo['denominator'],
                     locs=mo['locs'], n_subs=mo['n_subs'], meta=mo['meta'],
                     date_created=mo['date_created'])

    # load nifti
    elif fname.split('.')[-1]=='nii' or '.'.join(fname.split('.')[-2:])=='nii.gz':
        return load_nifti(fname)

def load_nifti(fname, mask_strategy='background'):
    """
    Load nifti file and convert to brain object
    """

    # load image
    img = nb.load(fname)

    # mask image
    mask = NiftiMasker(mask_strategy=mask_strategy)
    mask.fit(fname)

    # get header
    hdr = img.get_header()

    # get affine
    S = img.get_sform()

    # get voxel size
    vox_size = hdr.get_zooms()

    # get image shape
    im_size = img.shape

    #
    if len(img.shape) > 3:
        N = img.shape[3]
    else:
        N = 1

    Y = mask.transform(fname)
    V = Y.shape[1]
    vmask = np.nonzero(np.array(np.reshape(mask.mask_img_.dataobj, (1, np.prod(mask.mask_img_.shape)), order='C')))[1]
    vox_coords = fullfact(img.shape[0:3])[vmask, ::-1]-1

    locs = np.array(np.dot(vox_coords, S[0:3, 0:3])) + S[:3, 3]

    return Brain(data=Y, locs=locs, meta={'header' : hdr})


def fullfact(dims):
    '''
    Replicates MATLAB's fullfact function (behaves the same way)
    '''
    vals = np.asmatrix(range(1, dims[0] + 1)).T
    if len(dims) == 1:
        return vals
    else:
        aftervals = np.asmatrix(fullfact(dims[1:]))
        inds = np.asmatrix(np.zeros((np.prod(dims), len(dims))))
        row = 0
        for i in range(aftervals.shape[0]):
            inds[row:(row + len(vals)), 0] = vals
            inds[row:(row + len(vals)), 1:] = np.tile(aftervals[i, :], (len(vals), 1))
            row += len(vals)
        return inds


def model_compile(data):
    """
    Compile existing expanded correlation matrices.

    Parameters
    ----------

    data : list of model object file directories
        This is specific for data replication and probably shouldn't be in the package release

    Returns
    ----------

    model : Model object
        A new updated model object

    """
    m = load(data[0])
    numerator = m.numerator
    denominator = m.denominator
    n_subs = 1

    for mo in data[1:]:
        m = load(mo)
        #numerator = np.nansum(np.dstack((numerator, m.numerator)), 2)
        numerator += m.numerator
        denominator += m.denominator
        n_subs += 1

    return Model(numerator=numerator, denominator=denominator,
                 locs=m.locs, n_subs=n_subs)
    ### this concatenation of locations doesn't work when updating an existing model (but would be necessary for a build)
    # return Model(numerator=numerator, denominator=denominator,
    #              locs=pd.concat([m.locs, bo.locs]), n_subs=n_subs)
