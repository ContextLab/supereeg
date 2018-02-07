
Building a brain object
=======================

Brain objects are supereeg’s fundamental data structure for a single
subject’s iEEG data. To create one at minimum you’ll need a matrix of
neural recordings (time samples by electrodes), electrode locations, and
a sample rate. Additionally, you can include information about separate
recording sessions and store custom meta data. In this tutorial, we’ll
build a brain object from scratch and get familiar with some of the
methods.

Load in the required libraries
==============================

.. code:: ipython2

    import supereeg as se
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt


.. parsed-literal::

    /Library/Python/2.7/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)


Simulate some data
==================

First, we’ll use supereeg’s built in simulation functions to simulate
some data and electrodes. By default, the ``simualate_data`` function
will return a 1000 samples by 10 electrodes matrix, but you can specify
the number of time samples with ``n_samples`` and the number of
electrodes with ``n_elecs``. If you want further information on
simulating data, check out the simulate tutorial!

.. code:: ipython2

    # simulate some data
    data = se.simulate_bo(n_samples=1000, sessions=2, n_elecs=10).get_data()
    
    # plot it
    plt.plot(data)
    plt.xlabel('time samples')
    plt.ylabel('activation')
    plt.show()



.. image:: brain_objects_files/brain_objects_4_0.png


We’ll also simulate some electrode locations

.. code:: ipython2

    locs = se.simulate_locations()
    print(locs)


.. parsed-literal::

        x   y   z
    0 -32 -19 -21
    1 -41 -12  23
    2 -10 -30  -7
    3   4 -13 -10
    4 -14 -45   2
    5  -2 -35 -14
    6 -40  47 -29
    7 -45   6  13
    8 -14 -11 -25
    9 -31  45  11


Creating a brain object
=======================

To construct a new brain objects, simply pass the data and locations to
the ``Brain`` class like this:

.. code:: ipython2

    bo = se.Brain(data=data, locs=locs, sample_rate=100)

To view a summary of the contents of the brain object, you can call the
``info`` function:

.. code:: ipython2

    bo.info()


.. parsed-literal::

    Number of electrodes: 10
    Recording time in seconds: [ 10.]
    Number of sessions: 1
    Date created: Wed Feb  7 17:47:18 2018
    Meta data: {}


Optionally, you can pass a ``sessions`` parameter, which is can be a
numpy array or list the length of your data with a unique identifier for
each session. For example:

.. code:: ipython2

    sessions = np.array([1]*(data.shape[0]/2)+[2]*(data.shape[0]/2))
    bo = se.Brain(data=data, locs=locs, sample_rate=1000, sessions=sessions)
    bo.info()


.. parsed-literal::

    Number of electrodes: 10
    Recording time in seconds: [ 0.5  0.5]
    Number of sessions: 2
    Date created: Wed Feb  7 17:47:18 2018
    Meta data: {}


You can also pass add custom meta data to the brain object to help keep
track of its contents. ``meta`` is a dictionary comprised of whatever
you want:

.. code:: ipython2

    meta = {
        'subjectID' : '123',
        'Investigator' : 'Andy',
        'Hospital' : 'DHMC'
    }
    bo = se.Brain(data=data, locs=locs, sample_rate=1000, sessions=sessions, meta=meta)
    bo.info()


.. parsed-literal::

    Number of electrodes: 10
    Recording time in seconds: [ 0.5  0.5]
    Number of sessions: 2
    Date created: Wed Feb  7 17:47:18 2018
    Meta data: {'Hospital': 'DHMC', 'subjectID': '123', 'Investigator': 'Andy'}


The structure of a brain object
===============================

Inside the brain object, the iEEG data is stored as a Pandas DataFrame
that can be accessed directly:

.. code:: ipython2

    bo.data.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.198819</td>
          <td>-0.124699</td>
          <td>-0.380240</td>
          <td>-0.685776</td>
          <td>-0.171183</td>
          <td>0.533168</td>
          <td>-0.179924</td>
          <td>-0.236687</td>
          <td>-0.423308</td>
          <td>0.167720</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.788142</td>
          <td>-1.248452</td>
          <td>-0.820344</td>
          <td>-0.984107</td>
          <td>-1.212429</td>
          <td>-0.888530</td>
          <td>-1.292391</td>
          <td>-0.868692</td>
          <td>-1.550727</td>
          <td>-0.620401</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.500353</td>
          <td>-1.188048</td>
          <td>-0.682745</td>
          <td>-0.901951</td>
          <td>-0.087577</td>
          <td>-0.153879</td>
          <td>-0.546635</td>
          <td>-0.441206</td>
          <td>0.296296</td>
          <td>-0.350318</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.933650</td>
          <td>0.196506</td>
          <td>0.472087</td>
          <td>0.391212</td>
          <td>0.161351</td>
          <td>0.548689</td>
          <td>1.305987</td>
          <td>1.009641</td>
          <td>1.124480</td>
          <td>0.577910</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.928871</td>
          <td>0.549550</td>
          <td>0.611523</td>
          <td>-0.842358</td>
          <td>-0.356948</td>
          <td>0.733351</td>
          <td>-1.286813</td>
          <td>-0.270330</td>
          <td>-0.209234</td>
          <td>0.854060</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[  1.98819461e-01,  -1.24698598e-01,  -3.80239902e-01, ...,
             -2.36686565e-01,  -4.23308220e-01,   1.67719980e-01],
           [ -7.88142011e-01,  -1.24845174e+00,  -8.20343604e-01, ...,
             -8.68692345e-01,  -1.55072709e+00,  -6.20401315e-01],
           [ -5.00353492e-01,  -1.18804812e+00,  -6.82745092e-01, ...,
             -4.41206200e-01,   2.96296474e-01,  -3.50318367e-01],
           ..., 
           [  2.03874039e-01,  -1.38341973e-01,   2.66879442e-01, ...,
              9.86484805e-01,   6.36961862e-01,   5.42720508e-01],
           [ -4.42432477e-01,  -6.84193728e-01,  -4.95644717e-01, ...,
             -1.51241736e-01,  -3.66008465e-01,   2.18509737e-01],
           [ -5.01381101e-01,  -4.20362270e-01,   8.03228790e-04, ...,
             -3.32534910e-01,  -9.00252363e-02,  -3.73585975e-01]])



Similarly, the electrode locations are stored as a Pandas DataFrame, and
can be retrieved as a numpy array using the ``get_locs`` method:

.. code:: ipython2

    bo.locs.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>x</th>
          <th>y</th>
          <th>z</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-32</td>
          <td>-19</td>
          <td>-21</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-41</td>
          <td>-12</td>
          <td>23</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-10</td>
          <td>-30</td>
          <td>-7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>-13</td>
          <td>-10</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-14</td>
          <td>-45</td>
          <td>2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[-32, -19, -21],
           [-41, -12,  23],
           [-10, -30,  -7],
           [  4, -13, -10],
           [-14, -45,   2],
           [ -2, -35, -14],
           [-40,  47, -29],
           [-45,   6,  13],
           [-14, -11, -25],
           [-31,  45,  11]])



You can also pass a list of indices for either ``times`` or ``locs`` and
return a subset of the brain object

.. code:: ipython2

    bo_s = bo.get_slice(times=[1,2,3], locs=[1,2,3])
    bo_s.get_data()




.. parsed-literal::

    array([[-1.24845174, -0.8203436 , -0.98410739],
           [-1.18804812, -0.68274509, -0.90195075],
           [ 0.19650629,  0.47208747,  0.39121229]])



You can also plot both the data and the electrode locations:

.. code:: ipython2

    bo.plot_data()
    plt.show()


.. parsed-literal::

    /Users/lucyowen/repos/superEEG/supereeg/brain.py:298: MatplotlibDeprecationWarning: The set_axis_bgcolor function was deprecated in version 2.0. Use set_facecolor instead.
      ax.set_axis_bgcolor('w')



.. image:: brain_objects_files/brain_objects_25_1.png


.. code:: ipython2

    bo.plot_locs()


.. parsed-literal::

    /Library/Python/2.7/site-packages/matplotlib/cbook.py:136: MatplotlibDeprecationWarning: The axisbg attribute was deprecated in version 2.0. Use facecolor instead.
      warnings.warn(message, mplDeprecation, stacklevel=1)
    /Library/Python/2.7/site-packages/nilearn/plotting/glass_brain.py:164: MatplotlibDeprecationWarning: The get_axis_bgcolor function was deprecated in version 2.0. Use get_facecolor instead.
      black_bg = colors.colorConverter.to_rgba(ax.get_axis_bgcolor()) \
    /Library/Python/2.7/site-packages/nilearn/plotting/displays.py:1259: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if node_color == 'auto':



.. image:: brain_objects_files/brain_objects_26_1.png


The other pieces of the brain object are listed below:

.. code:: ipython2

    # array of session identifiers for each timepoint
    sessions = bo.sessions
    
    # number of sessions
    n_sessions = bo.n_sessions
    
    # sample rate
    sample_rate = bo.sample_rate
    
    # number of electrodes
    n_elecs = bo.n_elecs
    
    # length of each recording session in seconds
    n_seconds = bo.n_secs
    
    # the date and time that the bo was created
    date_created = bo.date_created
    
    # kurtosis of each electrode
    kurtosis = bo.kurtosis
    
    # meta data
    meta = bo.meta
    
    # label delinieating observed and reconstructed locations
    label = bo.label

Brain object methods
====================

There are a few other useful methods on a brain object

``bo.info()``
-------------

This method will give you a summary of the brain object:

.. code:: ipython2

    bo.info()


.. parsed-literal::

    Number of electrodes: 10
    Recording time in seconds: [ 0.5  0.5]
    Number of sessions: 2
    Date created: Wed Feb  7 17:47:18 2018
    Meta data: {'Hospital': 'DHMC', 'subjectID': '123', 'Investigator': 'Andy'}


``bo.get_data()``
-----------------

.. code:: ipython2

    data_array = bo.get_data()

``bo.get_zscore_data()``
------------------------

This method will return a numpy array of the zscored data:

.. code:: ipython2

    zdata_array = bo.get_zscore_data()

``bo.get_locs()``
-----------------

This method will return a numpy array of the electrode locations:

.. code:: ipython2

    locs = bo.get_locs()

``bo.plot_data()``
------------------

This method normalizes and plots data from brain object:

.. code:: ipython2

    bo.plot_data()



.. image:: brain_objects_files/brain_objects_39_0.png


``bo.plot_locs()``
------------------

This method plots electrode locations from brain object:

.. code:: ipython2

    bo.plot_locs()



.. image:: brain_objects_files/brain_objects_41_0.png


``bo.save(fname='something')``
------------------------------

This method will save the brain object to the specified file location.
The data will be saved as a ‘bo’ file, which is a dictionary containing
the elements of a brain object saved in the hd5 format using
``deepdish``.

.. code:: ipython2

    #bo.save(fname='brain_object')

``bo.to_nii()``
---------------

This method converts the brain object into a ``nibabel`` nifti image. If
``filepath`` is specified, the nifti file will be saved. You can also
specify a nifti template with the ``template`` argument.

.. code:: ipython2

    nii = bo.to_nii()
    print(type(nii))
    
    # save the file
    # nii = bo.to_nii(filepath='/path/to/file/brain')
    
    # specify a template
    # nii = bo.to_nii(template='/path/to/nifti/file.nii')


.. parsed-literal::

    <class 'nibabel.nifti1.Nifti1Image'>


.. parsed-literal::

    /Users/lucyowen/repos/superEEG/supereeg/brain.py:396: UserWarning: Voxel sizes of reconstruction and template do not match. Default to using a template with 20mm voxels.
      warnings.warn('Voxel sizes of reconstruction and template do not match. '
    /Users/lucyowen/repos/superEEG/supereeg/brain.py:417: UserWarning: Voxel sizes of reconstruction and template do not match. Voxel sizes calculated from model locations.
      warnings.warn('Voxel sizes of reconstruction and template do not match. '
    /Users/lucyowen/repos/superEEG/supereeg/brain.py:436: RuntimeWarning: invalid value encountered in divide
      data = np.divide(data, counts)

