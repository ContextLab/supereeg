
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

    import warnings 
    warnings.simplefilter("ignore")
    %matplotlib inline
    import supereeg as se
    import numpy as np

Simulate some data
==================

First, we’ll use supereeg’s built in simulation functions to simulate
some data and electrodes. By default, the ``simulate_data`` function
will return a 1000 samples by 10 electrodes matrix, but you can specify
the number of time samples with ``n_samples`` and the number of
electrodes with ``n_elecs``. If you want further information on
simulating data, check out the simulate tutorial!

.. code:: ipython2

    # simulate some data
    bo_data = se.simulate_bo(n_samples=1000, sessions=2, n_elecs=10)
    
    # plot it
    bo_data.plot_data()
    
    # get just data
    data = bo_data.get_data()



.. image:: brain_objects_files/brain_objects_4_0.png


We’ll also simulate some electrode locations

.. code:: ipython2

    locs = se.simulate_locations()
    print(locs)


.. parsed-literal::

        x   y   z
    0 -44 -50  20
    1 -41  28  -3
    2 -36  -8  36
    3 -21 -21 -23
    4 -14  40  12
    5 -11   0  -5
    6  -9 -45  12
    7  10   4 -41
    8  24  -9 -33
    9  43  31 -13


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
    Recording time in seconds: [10.]
    Sample Rate in Hz: [100]
    Number of sessions: 1
    Date created: Thu Mar  8 12:14:43 2018
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
    Recording time in seconds: [0.5 0.5]
    Sample Rate in Hz: [1000, 1000]
    Number of sessions: 2
    Date created: Thu Mar  8 12:14:43 2018
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
    Recording time in seconds: [0.5 0.5]
    Sample Rate in Hz: [1000, 1000]
    Number of sessions: 2
    Date created: Thu Mar  8 12:14:43 2018
    Meta data: {'Hospital': 'DHMC', 'subjectID': '123', 'Investigator': 'Andy'}


Initialize brain objects
========================

Brain objects can be initialized by passing a brain object (ending in
``.bo``), but can also be initialized with a model object or nifti
object by specifying ``return_type`` as ``bo`` in the load function.

For example, you can load a nifti object as a brain object:

.. code:: ipython2

    se.load('example_nifti', return_type='bo')




.. parsed-literal::

    <supereeg.brain.Brain at 0x10db64a90>



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
          <td>-0.164898</td>
          <td>0.449080</td>
          <td>-0.123422</td>
          <td>-0.487585</td>
          <td>-0.558142</td>
          <td>-0.068899</td>
          <td>-1.062847</td>
          <td>-0.598438</td>
          <td>-0.608100</td>
          <td>0.264699</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.158617</td>
          <td>0.195738</td>
          <td>0.283509</td>
          <td>0.309854</td>
          <td>0.256612</td>
          <td>0.042650</td>
          <td>0.392309</td>
          <td>0.243883</td>
          <td>0.385083</td>
          <td>0.228009</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.088609</td>
          <td>0.002041</td>
          <td>0.492744</td>
          <td>0.423460</td>
          <td>0.127684</td>
          <td>-0.080433</td>
          <td>0.112324</td>
          <td>0.598199</td>
          <td>0.314062</td>
          <td>0.057484</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.498890</td>
          <td>1.118350</td>
          <td>0.696200</td>
          <td>0.934301</td>
          <td>0.483936</td>
          <td>0.753088</td>
          <td>0.530552</td>
          <td>0.757954</td>
          <td>-0.010164</td>
          <td>0.690289</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-0.495082</td>
          <td>-0.797964</td>
          <td>-0.442069</td>
          <td>-0.238359</td>
          <td>-0.111551</td>
          <td>-0.685585</td>
          <td>-0.968015</td>
          <td>-0.255937</td>
          <td>-0.715173</td>
          <td>-0.107770</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[-0.16489751,  0.44908036, -0.12342183, ..., -0.59843798,
            -0.60810028,  0.26469928],
           [-0.15861694,  0.19573777,  0.28350922, ...,  0.2438828 ,
             0.38508338,  0.22800868],
           [-0.08860861,  0.00204105,  0.49274423, ...,  0.59819917,
             0.31406165,  0.05748386],
           ...,
           [-0.42121278, -0.4665349 , -0.17873552, ...,  0.94870931,
             0.52978121,  0.23011087],
           [-0.88400841, -0.79266786, -0.59817874, ..., -0.36894899,
            -0.1108448 , -0.60039107],
           [ 0.47997522,  0.28857125,  0.64570221, ..., -0.66169582,
            -0.03126307,  0.7099525 ]])



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
          <td>-44</td>
          <td>-50</td>
          <td>20</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-41</td>
          <td>28</td>
          <td>-3</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-36</td>
          <td>-8</td>
          <td>36</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-21</td>
          <td>-21</td>
          <td>-23</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-14</td>
          <td>40</td>
          <td>12</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[-44, -50,  20],
           [-41,  28,  -3],
           [-36,  -8,  36],
           [-21, -21, -23],
           [-14,  40,  12],
           [-11,   0,  -5],
           [ -9, -45,  12],
           [ 10,   4, -41],
           [ 24,  -9, -33],
           [ 43,  31, -13]])



You can also pass a list of indices for either ``times`` or ``locs`` and
return a subset of the brain object

.. code:: ipython2

    bo_s = bo.get_slice(sample_inds=[1,2,3], loc_inds=[1,2,3])
    bo_s.get_data()




.. parsed-literal::

    array([[0.19573777, 0.28350922, 0.30985394],
           [0.00204105, 0.49274423, 0.42346033],
           [1.11835032, 0.69620039, 0.93430131]])



You can resample your data by specifying a new resample rate

.. code:: ipython2

    bo.resample(64)

You can also plot both the data and the electrode locations:

.. code:: ipython2

    bo.plot_data()



.. image:: brain_objects_files/brain_objects_29_0.png


.. code:: ipython2

    bo.plot_locs()



.. image:: brain_objects_files/brain_objects_30_0.png


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
    Recording time in seconds: [0.5 0.5]
    Sample Rate in Hz: [64, 64]
    Number of sessions: 2
    Date created: Thu Mar  8 12:14:43 2018
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

``bo.get_slice()``
------------------

This method allows you to slice out time and locations from the brain
object, and returns a brain object. This can occur in place if you set
the flag ``inplace=True``.

.. code:: ipython2

    bo_slice = bo.get_slice(sample_inds=None, loc_inds=None, inplace=False)

``bo.resample()``
-----------------

This method allows you resample a brain object in place.

.. code:: ipython2

    bo.resample(resample_rate=None)




.. parsed-literal::

    <supereeg.brain.Brain at 0x10dd2a250>



``bo.plot_data()``
------------------

This method normalizes and plots data from brain object:

.. code:: ipython2

    bo.plot_data()



.. image:: brain_objects_files/brain_objects_47_0.png


``bo.plot_locs()``
------------------

This method plots electrode locations from brain object:

.. code:: ipython2

    bo.plot_locs()



.. image:: brain_objects_files/brain_objects_49_0.png


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
specify a nifti template with the ``template`` argument. If no template
is specified, it will use the gray matter masked MNI 152 brain
downsampled to 6mm.

.. code:: ipython2

    # convert to nifti
    nii = bo.to_nii()
    
    # plot first timepoint
    nii.plot_glass_brain()
    
    # save the file
    # nii = bo.to_nii(filepath='/path/to/file/brain')
    
    # specify a template and resolution
    # nii = bo.to_nii(template='/path/to/nifti/file.nii', vox_size=20)



.. image:: brain_objects_files/brain_objects_53_0.png

