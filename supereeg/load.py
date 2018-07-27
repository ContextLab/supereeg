from __future__ import print_function
import os
import warnings
import requests
import numpy as np
import deepdish as dd
from datetime import datetime
from .brain import Brain
from .model import Model
from .nifti import Nifti
from .location import Location
from .helpers import _resample_nii

BASE_URL = 'https://docs.google.com/uc?export=download'
homedir = os.path.expanduser('~')
datadir = os.path.join(homedir, 'supereeg_data')

datadict = { #TODO: do the data types need to be specified or could they be inferred from the downloaded objects?
    'example_data' : ['1kijSKt-QLEZ1O3J5Pk-8aByn33bPCAFl', 'bo'],
    'example_model' : ['1l4s7mE0KbPMmIcIA9JQzSZHCA8LWFq1I', 'mo'],
    'example_nifti' : ['17VeBTruexTERwBjq1UvU6BbBRt0jkBzi', 'nii'],
    'example_filter' : ['1eHcYg1idIK8y2LMLK_tqSxB7jI_l7OsL', 'bo'],
    'std' : ['1P-WcEBVYnoMQAYhvSCf1BBMIDMe9VZIM', 'nii'],
    'gray' : ['1a8wptBaMIFEl4j8TFhlTQVUAbyC0sN4p', 'nii'],
    'pyFR_k10r20_20mm' : ['1l4s7mE0KbPMmIcIA9JQzSZHCA8LWFq1I', 'mo'],
    'pyFR_k10r20_6mm' : ['1yH47fldoeuK0AQtOhMM-P2P0Dv_zH5G6', 'mo']
}

def load(fname, vox_size=None, return_type=None, sample_inds=None,
         loc_inds=None, field=None):
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


        Examples include :

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

    sample_inds : int, list or slice
        Indices of samples you'd like to load in. Only works for Brain object.

    loc_inds : int, list or slice
        Indices of slices you'd like to load in. Only works for Brain object.

    field : str
        The particular field of the data you want to load. This will work for
        Brain objects and Model objects.

    Returns
    ----------
    data : supereeg.Nifti, supereeg.Brain or supereeg.Model
        Data to be returned

    """
    if field != None and (sample_inds!=None or loc_inds!=None):
        raise ValueError("Using both field and slicing currently not supported.")

    if fname in datadict.keys():
        data = _load_example(fname, datadict[fname], sample_inds, loc_inds, field)
    else:
        data = _load_from_path(fname, sample_inds, loc_inds, field)
    if field is None:
        return _convert(data, return_type, vox_size)
    else:
        return data

def _convert(data, return_type, vox_size):
    """ Converts between bo, mo and nifti """
    if return_type is None and vox_size is None:
        return data
    elif return_type is None and vox_size is not None:
        if type(data) is Nifti:
            return _resample_nii(data, target_res=vox_size)
        else:
            warnings.warn('Data is not a Nifti file, therefore vox_size was not computed '
                          ' Please specify nii as return_type if you would like a Nifti returned.')
            return data
    elif return_type is 'nii':
        if type(data) is not Nifti:
            data = Nifti(data)
        if vox_size:
            return _resample_nii(data, target_res=vox_size)
        else:
            return data
    elif return_type is 'bo':
        if type(data) is not Brain:
            data = Brain(data)
        return data
    elif return_type is 'mo':
        if type(data) is not Model:
            data = Model(data)
        return data

def _load_example(fname, fileid, sample_inds, loc_inds, field):
    """ Loads in dataset given a google file id """
    fullpath = os.path.join(homedir, 'supereeg_data', fname + '.' + fileid[1])
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if not os.path.exists(fullpath):
        try:
            _download(fname, _load_stream(fileid[0]), fileid[1])
            data = _load_from_cache(fname, fileid[1], sample_inds, loc_inds, field)
        except ValueError as e:
            print(e)
            raise ValueError('Download failed.')
    else:
        try:
            data = _load_from_cache(fname, fileid[1], sample_inds, loc_inds, field)
        except:
            try:
                _download(fname, _load_stream(fileid[0]), fileid[1])
                data = _load_from_cache(fname, fileid[1], sample_inds, loc_inds, field)
            except ValueError as e:
                print(e)
                raise ValueError('Download failed. Try deleting cache data in'
                                 ' /Users/homedir/supereeg_data.') #FIXME: use generic home directory reference rather than platform-specific path
    return data

def _load_stream(fileid):
    """ Retrieve data from google drive """
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    url = BASE_URL + fileid
    session = requests.Session()
    response = session.get(BASE_URL, params = { 'id' : fileid }, stream = True)
    token = _get_confirm_token(response)
    if token:
        params = { 'id' : fileid, 'confirm' : token }
        response = session.get(BASE_URL, params = params, stream = True)
    return response

def _download(fname, data, ext):
    """ Download data to cache """
    fullpath = os.path.join(homedir, 'supereeg_data', fname)
    with open(fullpath + '.' + ext, 'wb') as f:
        f.write(data.content)

def _load_from_path(fpath, sample_inds=None, loc_inds=None, field=None):
    """ Load a file from a local path """
    try:
        ext = fpath.split('.')[-1]
    except:
        raise ValueError("Must specify a file extension.")
    if field != None:
        if ext in ['bo', 'mo']:
            return _load_field(fpath, field)
        else:
            raise ValueError("Can only load field from Brain or Model object.")
    elif ext=='bo':
        if sample_inds!=None or loc_inds!=None:
            return Brain(**_load_slice(fpath, sample_inds, loc_inds))
        else:
            return Brain(**dd.io.load(fpath))
    elif ext=='mo':
        return Model(**dd.io.load(fpath))
    elif ext in ('nii', 'gz'):
        return Nifti(fpath)
    else:
        raise ValueError("Filetype not recognized. Must be .bo, .mo or .nii.")

def _load_from_cache(fname, ftype, sample_inds=None, loc_inds=None, field=None):
    """ Load a file from local data cache """
    fullpath = os.path.join(homedir, 'supereeg_data', fname + '.' + ftype)
    if field != None:
        if ftype in ['bo', 'mo']:
            return _load_field(fullpath, field)
        else:
            raise ValueError("Can only load field from Brain or Model object.")
    elif ftype is 'bo':
        if sample_inds!=None or loc_inds!=None:
            return Brain(**_load_slice(fullpath, sample_inds, loc_inds))
        else:
            return Brain(**dd.io.load(fullpath))
    elif ftype is 'mo':
        # if the model was created using supereeg<0.2.0, load using the "old" format
        # (i.e. supereeg>=0.2.0 computes model in log space)
        date_created = _load_field(fullpath, field='date_created')
        if datetime.strptime(date_created, "%c")< datetime(2018, 7, 27, 14, 40, 48, 359141):
            num = _load_field(fullpath, field='numerator')
            den = _load_field(fullpath, field='denominator')
            locs = _load_field(fullpath, field='locs')
            n_subs = _load_field(fullpath, field='n_subs')
            return Model(data=np.divide(num, den), locs=locs, n_subs=n_subs)
        else:
            return Model(**dd.io.load(fullpath))
    elif ftype is 'nii':
        return Nifti(fullpath)
    elif ftype is 'locs':
        return Location(fullpath)

def _load_field(fname, field):
    """ Loads a particular field of a file """
    return dd.io.load(fname, group='/' + field) #FIXME: use os.path.join rather than using slashes

def _load_slice(fname, sample_inds=None, loc_inds=None):
    """
    Load a slice of a brain object

    Parameters
    ----------
    fname : str
        Path to brain object

    sample_inds : int, list or slice
        Indices of samples you'd like to load in

    loc_inds : int, list or slice
        Indices of slices you'd like to load in

    Returns
    ----------
    data : dict
        Dictionary of contents to pass to brain object

    """

    sr = dd.io.load(fname, group='/sample_rate') #FIXME: use os.path.join rather than using slashes
    meta = dd.io.load(fname, group='/meta') #FIXME: use os.path.join rather than using slashes
    date_created = dd.io.load(fname, group='/date_created') #FIXME: use os.path.join rather than using slashes

    if sample_inds!=None and loc_inds!=None:
        if not isinstance(sample_inds, int) and not isinstance(loc_inds, int):
            raise IndexError("Slicing with 2 lists is currently not supported.") #FIXME: make this message more specific
        data = dd.io.load(fname, group='/data', sel=dd.aslice[sample_inds, loc_inds]) #FIXME: use os.path.join rather than using slashes
        locs = dd.io.load(fname, group='/locs', sel=dd.aslice[loc_inds, :]) #FIXME: use os.path.join rather than using slashes
        sessions = dd.io.load(fname, group='/sessions').iloc[sample_inds].tolist() #FIXME: use os.path.join rather than using slashes
    elif loc_inds==None:
        data = dd.io.load(fname, group='/data', sel=dd.aslice[sample_inds, :]) #FIXME: use os.path.join rather than using slashes
        locs = dd.io.load(fname, group='/locs') #FIXME: use os.path.join rather than using slashes
        sessions = dd.io.load(fname, group='/sessions').iloc[sample_inds].tolist() #FIXME: use os.path.join rather than using slashes
    elif sample_inds==None:
        data = dd.io.load(fname, group='/data', sel=dd.aslice[:, loc_inds]) #FIXME: use os.path.join rather than using slashes
        locs = dd.io.load(fname, group='/locs', sel=dd.aslice[loc_inds, :]) #FIXME: use os.path.join rather than using slashes
        sessions = dd.io.load(fname, group='/sessions').tolist() #FIXME: use os.path.join rather than using slashes
    sample_rate = [sr[int(s-1)] for s in np.unique(sessions)]
    data = np.atleast_2d(data)
    locs = np.atleast_2d(locs)
    if locs.shape[0]==1:
        if data.shape[1]>data.shape[0]:
            data = data.T
    return dict(data=data, locs=locs,
                sample_rate=sample_rate, meta=meta, date_created=date_created)
