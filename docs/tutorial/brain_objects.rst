
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
    0 -41 -32 -25
    1  14  25  33
    2  16 -40 -40
    3  18 -46  -7
    4  23 -39   0
    5  26 -50 -12
    6  31 -12 -31
    7  38 -46   7
    8  46  13  28
    9  47  38  16


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
    Date created: Thu Mar  8 10:31:32 2018
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
    Date created: Thu Mar  8 10:31:32 2018
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
    Date created: Thu Mar  8 10:31:32 2018
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

    <supereeg.brain.Brain at 0x10d6de290>



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
          <td>1.440724</td>
          <td>2.600472</td>
          <td>1.616553</td>
          <td>0.326513</td>
          <td>0.065006</td>
          <td>0.826129</td>
          <td>-0.269940</td>
          <td>0.560531</td>
          <td>0.826667</td>
          <td>1.265671</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.785873</td>
          <td>0.343442</td>
          <td>1.099341</td>
          <td>0.971345</td>
          <td>0.615809</td>
          <td>0.463145</td>
          <td>0.388826</td>
          <td>0.758021</td>
          <td>0.595034</td>
          <td>0.737589</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.385723</td>
          <td>-0.841534</td>
          <td>-0.513777</td>
          <td>-0.978683</td>
          <td>-0.891984</td>
          <td>-0.928374</td>
          <td>-1.272700</td>
          <td>-1.118253</td>
          <td>-0.961862</td>
          <td>-0.619235</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.939826</td>
          <td>1.013013</td>
          <td>0.754051</td>
          <td>-0.216940</td>
          <td>0.478102</td>
          <td>1.282567</td>
          <td>0.036766</td>
          <td>0.222427</td>
          <td>0.574123</td>
          <td>1.085053</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.786556</td>
          <td>0.115281</td>
          <td>0.932562</td>
          <td>0.811978</td>
          <td>0.447603</td>
          <td>0.619423</td>
          <td>0.784246</td>
          <td>0.949721</td>
          <td>0.601160</td>
          <td>1.124628</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[ 1.44072358,  2.60047167,  1.61655288, ...,  0.56053092,
             0.82666728,  1.26567137],
           [ 0.78587303,  0.34344197,  1.09934086, ...,  0.75802063,
             0.59503439,  0.73758909],
           [-0.38572279, -0.84153441, -0.51377679, ..., -1.11825266,
            -0.96186171, -0.619235  ],
           ...,
           [-0.57304416, -1.23518946,  0.01491523, ...,  0.76143582,
             0.34894194,  0.45427171],
           [-1.51967449, -0.30406692, -1.55499431, ..., -1.17967999,
            -1.15453534, -1.1889883 ],
           [-0.01769448,  0.04239495, -0.17630018, ..., -0.54329545,
            -0.75459267, -0.25615349]])



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
          <td>-41</td>
          <td>-32</td>
          <td>-25</td>
        </tr>
        <tr>
          <th>1</th>
          <td>14</td>
          <td>25</td>
          <td>33</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16</td>
          <td>-40</td>
          <td>-40</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18</td>
          <td>-46</td>
          <td>-7</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23</td>
          <td>-39</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[-41, -32, -25],
           [ 14,  25,  33],
           [ 16, -40, -40],
           [ 18, -46,  -7],
           [ 23, -39,   0],
           [ 26, -50, -12],
           [ 31, -12, -31],
           [ 38, -46,   7],
           [ 46,  13,  28],
           [ 47,  38,  16]])



You can also pass a list of indices for either ``times`` or ``locs`` and
return a subset of the brain object

.. code:: ipython2

    bo_s = bo.get_slice(sample_inds=[1,2,3], loc_inds=[1,2,3])
    bo_s.get_data()




.. parsed-literal::

    array([[ 0.34344197,  1.09934086,  0.97134502],
           [-0.84153441, -0.51377679, -0.97868313],
           [ 1.01301269,  0.75405123, -0.21694023]])



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
    Date created: Thu Mar  8 10:31:32 2018
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

    <supereeg.brain.Brain at 0x10e8b6250>



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

