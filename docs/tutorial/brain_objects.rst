
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
    import supereeg as se
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

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
    data = se.simulate_bo(n_samples=1000, sessions=2, n_elecs=10).get_data()
    
    # plot it
    plt.plot(data)
    plt.xlabel('time samples')
    plt.ylabel('activation')
    plt.show()

We’ll also simulate some electrode locations

.. code:: ipython2

    locs = se.simulate_locations()
    print(locs)


.. parsed-literal::

        x   y   z
    0  -1   0 -42
    1  11 -11 -41
    2 -25  -2  48
    3  25  46 -17
    4   4  24 -26
    5  -6 -25  -8
    6   7  44  33
    7 -50 -12  21
    8  -9  34 -38
    9   7  18 -26


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
    Sample Rate in Hz: [100]
    Number of sessions: 1
    Date created: Fri Feb 16 12:36:39 2018
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
    Sample Rate in Hz: [1000, 1000]
    Number of sessions: 2
    Date created: Fri Feb 16 12:36:39 2018
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
    Sample Rate in Hz: [1000, 1000]
    Number of sessions: 2
    Date created: Fri Feb 16 12:36:39 2018
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
          <td>0.604922</td>
          <td>1.250532</td>
          <td>0.818492</td>
          <td>0.692866</td>
          <td>0.758488</td>
          <td>0.849834</td>
          <td>0.386241</td>
          <td>0.721922</td>
          <td>0.279655</td>
          <td>0.736418</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.082074</td>
          <td>0.645032</td>
          <td>0.231335</td>
          <td>-0.318368</td>
          <td>0.000778</td>
          <td>-0.188568</td>
          <td>0.091102</td>
          <td>-0.444249</td>
          <td>0.114340</td>
          <td>-0.562943</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.221214</td>
          <td>-0.320411</td>
          <td>-0.101885</td>
          <td>0.261285</td>
          <td>0.332768</td>
          <td>-0.215834</td>
          <td>0.170547</td>
          <td>0.510624</td>
          <td>0.359748</td>
          <td>-0.163719</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-0.657911</td>
          <td>-0.779742</td>
          <td>-0.858455</td>
          <td>-0.825788</td>
          <td>-1.667429</td>
          <td>-1.089516</td>
          <td>-0.856549</td>
          <td>-0.977772</td>
          <td>-0.991612</td>
          <td>-0.778056</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.225031</td>
          <td>1.078450</td>
          <td>0.848490</td>
          <td>0.660335</td>
          <td>0.595238</td>
          <td>1.359051</td>
          <td>0.916940</td>
          <td>0.700657</td>
          <td>0.826554</td>
          <td>0.877658</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[ 0.60492209,  1.25053182,  0.8184924 , ...,  0.72192165,
             0.27965501,  0.73641844],
           [-0.08207373,  0.64503197,  0.23133506, ..., -0.44424927,
             0.11434013, -0.56294302],
           [-0.22121389, -0.32041086, -0.10188461, ...,  0.51062402,
             0.35974763, -0.163719  ],
           ..., 
           [ 0.08948554,  0.36467453, -0.1590011 , ...,  0.68056159,
             0.67269439, -0.06838542],
           [-0.03670738,  0.84424327, -0.06850471, ..., -0.97074283,
            -1.055871  , -0.15987082],
           [ 0.29715632,  0.44001621,  0.01674216, ...,  1.08037481,
             0.55886292,  0.65656492]])



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
          <td>-1</td>
          <td>0</td>
          <td>-42</td>
        </tr>
        <tr>
          <th>1</th>
          <td>11</td>
          <td>-11</td>
          <td>-41</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-25</td>
          <td>-2</td>
          <td>48</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25</td>
          <td>46</td>
          <td>-17</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4</td>
          <td>24</td>
          <td>-26</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[ -1,   0, -42],
           [ 11, -11, -41],
           [-25,  -2,  48],
           [ 25,  46, -17],
           [  4,  24, -26],
           [ -6, -25,  -8],
           [  7,  44,  33],
           [-50, -12,  21],
           [ -9,  34, -38],
           [  7,  18, -26]])



You can also pass a list of indices for either ``times`` or ``locs`` and
return a subset of the brain object

.. code:: ipython2

    bo_s = bo.get_slice(times=[1,2,3], locs=[1,2,3])
    bo_s.get_data()




.. parsed-literal::

    array([[ 0.64503197,  0.23133506, -0.31836793],
           [-0.32041086, -0.10188461,  0.26128456],
           [-0.77974196, -0.85845497, -0.82578811]])



You can resample your data by specifying a new resample rate

.. code:: ipython2

    bo.resample(64)

You can also plot both the data and the electrode locations:

.. code:: ipython2

    bo.plot_data()
    plt.show()

.. code:: ipython2

    bo.plot_locs()

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
    Sample Rate in Hz: [64, 64]
    Number of sessions: 2
    Date created: Fri Feb 16 12:36:39 2018
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
object, and returns a brain object.

.. code:: ipython2

    bo_slice = bo.get_slice(times=None, locs=None)

``bo.resample()``
-----------------

This method allows you resample a brain object in place.

.. code:: ipython2

    bo.resample(resample_rate=None)




.. parsed-literal::

    <supereeg.brain.Brain at 0x10cb9d310>



``bo.plot_data()``
------------------

This method normalizes and plots data from brain object:

.. code:: ipython2

    bo.plot_data()

``bo.plot_locs()``
------------------

This method plots electrode locations from brain object:

.. code:: ipython2

    bo.plot_locs()

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

    # convert to nifit
    # nii = bo.to_nii()
    
    # save the file
    # nii = bo.to_nii(filepath='/path/to/file/brain')
    
    # specify a template
    # nii = bo.to_nii(template='/path/to/nifti/file.nii')

