from __future__ import print_function
import os
import pickle
import warnings
import numpy as np
import deepdish as dd
import pandas as pd
from .brain import Brain
from .model import Model
from .nifti import Nifti
from .helpers import tal2mni, _gray, _std, _resample_nii

def load(fname, vox_size=None, return_type=None):
    """
    Load nifti file, brain or model object, or example data.

    This function can load in example data, as well as nifti objects (.nii), brain objects (.bo)
    and model objects (.mo) by detecting the extension and calling the appropriate
    load function.  Thus, be sure to include the file extension in the fname
    parameter.

    Parameters
    ----------
    fname : str

        The name of the example data or a filepath.


        Examples includes :

            example_data - example brain object (n = 64)

            example_filter - load example patient data with kurtosis thresholded channels (n = 40)

            example_model - example model object with locations from gray masked brain downsampled to 20mm (n = 210)

            example_nifti - example nifti file from gray masked brain downsampled to 20mm (n = 210)


        Nifti templates :

            gray - load gray matter masked MNI 152 brain

            std - load MNI 152 standard brain


        Models :

            pyfr - model used for analyses from Owen LLW and Manning JR (2017) Towards Human Super EEG. bioRxiv: 121020`

                vox_size options: 6mm and 20mm


    vox_size : int or float

        Voxel size for loading and resampling nifti image

    return_type : str

        Option for loading data

            'bo' - returns supereeg.Brain

            'mo' - returns supereeg.Model

            'nii' - returns supereeg.Nifti



    Returns
    ----------
    data : supereeg.Nifti, supereeg.Brain or supereeg.Model
        Data to be returned

    """
    global loaded

    if type(fname) is str:

        if fname is 'example_data':
            with open(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/CH003.npz', 'rb') as handle:
                f = np.load(handle)
                data = f['Y']
                sample_rate = f['samplerate']
                sessions = f['fname_labels']
                locs = tal2mni(f['R'])
                meta = {'patient':'CH003'}

            loaded = Brain(data=data, locs=locs, sessions=sessions, sample_rate=sample_rate, meta=meta)

        elif fname is 'example_model':
            try:
                with open(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/example_model.mo',
                          'rb') as handle:
                    example_model = pickle.load(handle)
                loaded = example_model

            except:
                try:
                    mo = dd.io.load(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/example_model.mo')
                    loaded = Model(numerator=mo['numerator'], denominator=mo['denominator'],
                                 locs=mo['locs'], n_subs=mo['n_subs'], meta=mo['meta'],
                                 date_created=mo['date_created'])
                except:
                    model = pd.read_pickle(
                        os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/example_model.mo')
                    loaded = model

        # load example nifti
        elif fname is 'example_nifti':
            nii = _gray(20)
            loaded = nii

        # load example patient data with kurtosis thresholded channels
        elif fname is 'example_filter':
            bo = dd.io.load(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/example_filter.bo')
            loaded = Brain(data=bo['data'], locs=bo['locs'], sessions=bo['sessions'],
                         sample_rate=bo['sample_rate'], meta=bo['meta'],
                         date_created=bo['date_created'])

        # load union of pyFR electrode locations
        elif fname is 'pyFR_union':
            with open(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/pyFR_k10_locs.npz',
                      'rb') as handle:
                data = np.load(handle)
                locs = data['locs']
                print(('subjects = ', data['subjs']))
            return locs

            ## need this model still:

        # elif fname is 'pyFR':
        #     try:
        #         with open(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/gray_mask_6mm_brain.mo', 'rb') as handle:
        #             example_model = pickle.load(handle)
        #         return example_model
        #     except:
        #         try:
        #             mo = dd.io.load(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/gray_mask_6mm_brain.mo')
        #             return Model(numerator=mo['numerator'], denominator=mo['denominator'],
        #                          locs=mo['locs'], n_subs=mo['n_subs'], meta=mo['meta'],
        #                          date_created=mo['date_created'])
        #         except:
        #             model = pd.read_pickle(os.path.dirname(os.path.abspath(__file__)) + '/../supereeg/data/gray_mask_6mm_brain.mo')
        #             return model

        # load MNI 152 standard brain
        elif fname is 'std':
            if vox_size:
                std = _std(vox_size)
            else:
                std = _std()
            loaded = std

        # load gray matter masked MNI 152 brain
        elif fname is 'gray':

            if vox_size:
                gray = _gray(vox_size)
            else:
                gray = _gray()
            loaded = gray

        # load brain object
        elif fname.split('.')[-1] == 'bo':
            bo = dd.io.load(fname)
            loaded = Brain(data=bo['data'], locs=bo['locs'], sessions=bo['sessions'],
                         sample_rate=bo['sample_rate'], meta=bo['meta'],
                         date_created=bo['date_created'])

        # load model object
        elif fname.split('.')[-1] == 'mo':
            mo = dd.io.load(fname)
            loaded = Model(numerator=mo['numerator'], denominator=mo['denominator'],
                         locs=mo['locs'], n_subs=mo['n_subs'], meta=mo['meta'],
                         date_created=mo['date_created'])

        # load nifti
        elif fname.split('.')[-1] == 'nii' or '.'.join(fname.split('.')[-2:]) == 'nii.gz':
            loaded = Nifti(fname)


    else:
        loaded = fname

    assert isinstance(loaded, (Brain, Model, Nifti))

    if return_type == 'nii':

        if not type(loaded) is Nifti:
            loaded = Nifti(loaded)

        if vox_size:
            return _resample_nii(loaded, target_res=vox_size)

        else:
            return loaded

    elif return_type == 'bo':

        if not type(loaded) is Brain:
            loaded = Brain(loaded)

        return loaded

    elif return_type == 'mo':

        if not type(loaded) is Model:
            loaded = Model(loaded)

        return loaded

    elif return_type is None:

        return loaded

    else:
        warnings.warn('return_type not understood')
