from __future__ import print_function
import os
import pickle
import warnings
import numpy as np
import deepdish as dd
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker
from .brain import Brain
from .model import Model
from .helpers import tal2mni, _fullfact, _gray, _std

def load(fname):
    """
    Load nifti file, brain or model object, or example data.

    This function can load in example data, as well as nifti files, brain objects (.bo)
    and model objects (.mo) by detecting the extension and calling the appropriate
    load function.  Thus, be sure to include the file extension in the fname
    parameter.

    Parameters
    ----------
    fname : string
        The name of the example data or a filepath.

        Example data includes:

        example_data - example brain object (n = 64)

        example_model - example model object with locations from gray masked brain downsampled to 20mm (n = 210)

        example_locations - example location from gray masked brain downsampled to 20mm (n = 210)

        example_nifti - example nifti file from gray masked brain downsampled to 20mm (n = 210)

        ### need to recompute these with the new 210 locations
        pyFR_k10r20_20mm - model used for analyses from
        `Owen LLW and Manning JR (2017) Towards Human Super EEG.  bioRxiv: 121020` with 20mm resolution (n = 170)
        
        pyFR_k10r20_6mm - model used for analyses from
        `Owen LLW and Manning JR (2017) Towards Human Super EEG.  bioRxiv: 121020` with 6 mm resolution (n = 10K)

    Returns
    ----------
    data : nibabel.Nifti1, supereeg.Brain or supereeg.Model
        Data to be returned

    """

    if fname is 'example_data':
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/CH003.npz', 'rb') as handle:
            f = np.load(handle)
            data = f['Y']
            sample_rate = f['samplerate']
            sessions = f['fname_labels']
            locs = tal2mni(f['R'])
            meta = 'CH003'

        return Brain(data=data, locs=locs, sessions=sessions, sample_rate=sample_rate, meta= meta)

    elif fname is 'example_model':
        try:
            with open(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/mini_model.mo', 'rb') as handle:
                example_model = pickle.load(handle)
            return example_model

        except:
            try:
                mo = dd.io.load(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/mini_model.mo')
                return Model(numerator=mo['numerator'], denominator=mo['denominator'],
                             locs=mo['locs'], n_subs=mo['n_subs'], meta=mo['meta'],
                             date_created=mo['date_created'])
            except:
                model = pd.read_pickle(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/mini_model.mo')
                return model


    elif fname is 'pyFR_k10r20_20mm':
        try:
            with open(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/mini_model.mo', 'rb') as handle:
                example_model = pickle.load(handle)
            return example_model
        except:
            try:
                mo = dd.io.load(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/mini_model.mo')
                return Model(numerator=mo['numerator'], denominator=mo['denominator'],
                             locs=mo['locs'], n_subs=mo['n_subs'], meta=mo['meta'],
                             date_created=mo['date_created'])
            except:
                model = pd.read_pickle(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/mini_model.mo')
                return model


    elif fname is 'pyFR_k10r20_6mm':
        try:
            with open(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/gray_mask_6mm_brain.mo', 'rb') as handle:
                example_model = pickle.load(handle)
            return example_model
        except:
            try:
                mo = dd.io.load(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/gray_mask_6mm_brain.mo')
                return Model(numerator=mo['numerator'], denominator=mo['denominator'],
                             locs=mo['locs'], n_subs=mo['n_subs'], meta=mo['meta'],
                             date_created=mo['date_created'])
            except:
                model = pd.read_pickle(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/gray_mask_6mm_brain.mo')
                return model


    elif fname is 'example_locations':
        bo = get_brain_object(_gray(20))
        return bo.get_locs()

    # load example nifti
    elif fname is 'example_nifti':
        nii = _std(20)
        return nii

    # load example patient data with kurtosis thresholded channels
    elif fname is 'example_filter':
        bo = dd.io.load(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/example_filter.bo')
        return Brain(data=bo['data'], locs=bo['locs'], sessions=bo['sessions'],
                    sample_rate=bo['sample_rate'], meta=bo['meta'],
                    date_created=bo['date_created'])

    # load union of pyFR electrode locations
    elif fname is 'pyFR_union':
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/pyFR_k10_locs.npz', 'rb') as handle:
            data = np.load(handle)
            locs = data['locs']
            print(('subjects = ', data['subjs']))
        return locs

    # load gray matter masked MNI 152 brain downsampled to 20mm voxels
    elif fname is 'gray_mask_20mm_brain':
        bo = get_brain_object(_gray(20))
        return bo

    # load gray matter masked MNI 152 brain downsampled to 6mm voxels
    elif fname is 'gray_mask_6mm_brain':
        bo = get_brain_object( _gray(6))
        return bo

    # load MNI 152 standard brain
    elif fname is 'std':
        bo = get_brain_object( _std())
        return bo

    # load gray matter masked MNI 152 brain
    elif fname is 'gray':
        bo = get_brain_object( _gray())
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
        return get_brain_object(fname)


# def load_nifti(nifti_file, mask_file=None):
#     """
#     Load nifti file and convert to brain object
#
#     Parameters
#     ----------
#     fname : string
#         Filepath to nifti file.
#
#
#     Returns
#     ----------
#     results : brain object
#         Brain object from nifti file
#
#     """
#
#
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#
#         img = nib.load(nifti_file)
#         mask = NiftiMasker(mask_strategy='background')
#         if mask_file is None:
#             mask.fit(nifti_file)
#         else:
#             mask.fit(mask_file)
#
#     hdr = img.header
#     S = img.get_sform()
#     _vox_size = hdr.get_zooms()
#     im_size = img.shape
#
#     if len(img.shape) > 3:
#         N = img.shape[3]
#     else:
#         N = 1
#
#     Y = np.float64(mask.transform(nifti_file)).copy()
#     vmask = np.nonzero(np.array(np.reshape(mask.mask_img_.dataobj, (1, np.prod(mask.mask_img_.shape)), order='C')))[1]
#     vox_coords = _fullfact(img.shape[0:3])[vmask, ::-1]-1
#
#     R = np.array(np.dot(vox_coords, S[0:3, 0:3])) + S[:3, 3]
#
#     return Brain(data=Y, locs=R, meta={'header': hdr})


def get_brain_object(nifti, mask_file=None, precision=4):
    """
    Function that returns a brain object

    Parameters
    ----------
    x : str or nifti image

        If x is a nifti filepath, loads nifti and returns brain object

        If x is a nifti image, it returns a brain object

    Returns
    ----------
    results: brain object


    """

    if type(nifti) is nib.nifti1.Nifti1Image:
        img = nifti

    elif type(nifti) is str:
        if os.path.exists(nifti):
            img = nib.load(nifti)
        else:
            warnings.warn('Nifti format not supported')
    else:
        warnings.warn('Nifti format not supported')

    mask = NiftiMasker(mask_strategy='background')
    if mask_file is None:
        mask.fit(nifti)
    else:
        mask.fit(mask_file)

    hdr = img.header
    S = img.get_sform()

    Y = np.float64(mask.transform(nifti)).copy()
    vmask = np.nonzero(np.array(np.reshape(mask.mask_img_.dataobj, (1, np.prod(mask.mask_img_.shape)), order='C')))[1]
    vox_coords = _fullfact(img.shape[0:3])[vmask, ::-1] - 1

    R = np.array(np.dot(vox_coords, S[0:3, 0:3])) + S[:3, 3]

    return Brain(data=Y, locs=R, meta={'header': hdr})