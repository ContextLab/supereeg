
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
    0  28  13  45
    1 -50  30  13
    2  35 -17  47
    3 -29  -2 -47
    4 -48 -24  45
    5 -12 -45 -12
    6  17  -4 -24
    7  40 -45 -42
    8   3  32  33
    9  36 -40  -6


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
    Date created: Fri Feb 16 13:41:46 2018
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
    Date created: Fri Feb 16 13:41:46 2018
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
    Date created: Fri Feb 16 13:41:46 2018
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
          <td>0.346717</td>
          <td>0.310505</td>
          <td>-0.165949</td>
          <td>-1.701957</td>
          <td>-1.361027</td>
          <td>-0.454753</td>
          <td>-1.366771</td>
          <td>-1.456691</td>
          <td>-0.553106</td>
          <td>-0.291857</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.599660</td>
          <td>0.787613</td>
          <td>0.105329</td>
          <td>0.679863</td>
          <td>0.704405</td>
          <td>1.014001</td>
          <td>0.652765</td>
          <td>0.901722</td>
          <td>0.883272</td>
          <td>0.120943</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.121934</td>
          <td>1.104040</td>
          <td>1.488016</td>
          <td>1.298971</td>
          <td>1.441645</td>
          <td>1.268965</td>
          <td>0.723432</td>
          <td>1.684950</td>
          <td>1.300593</td>
          <td>0.943263</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-1.361016</td>
          <td>-1.490193</td>
          <td>-1.307363</td>
          <td>-1.375267</td>
          <td>-1.819020</td>
          <td>-1.657349</td>
          <td>-0.830506</td>
          <td>-1.272115</td>
          <td>-1.255302</td>
          <td>-1.488073</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-0.529114</td>
          <td>-0.501800</td>
          <td>-0.734973</td>
          <td>-1.288213</td>
          <td>-0.887256</td>
          <td>-0.343574</td>
          <td>-1.037217</td>
          <td>-1.033174</td>
          <td>-0.912046</td>
          <td>-0.646481</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[ 0.34671733,  0.31050466, -0.1659486 , ..., -1.45669071,
            -0.55310564, -0.29185711],
           [ 0.59965954,  0.78761269,  0.1053287 , ...,  0.9017221 ,
             0.88327153,  0.12094327],
           [ 1.12193434,  1.1040404 ,  1.48801596, ...,  1.68495037,
             1.30059271,  0.94326328],
           ..., 
           [ 0.10249368, -0.51581856,  0.04707269, ...,  0.00965142,
             0.05627562, -0.1317021 ],
           [ 0.12922342,  0.34001253, -0.29569431, ...,  0.13134371,
            -0.45341473, -0.11569072],
           [-0.23267049, -0.33454169, -0.13614004, ..., -0.45006763,
            -0.79275953, -0.04140593]])



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
          <td>28</td>
          <td>13</td>
          <td>45</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-50</td>
          <td>30</td>
          <td>13</td>
        </tr>
        <tr>
          <th>2</th>
          <td>35</td>
          <td>-17</td>
          <td>47</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-29</td>
          <td>-2</td>
          <td>-47</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-48</td>
          <td>-24</td>
          <td>45</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[ 28,  13,  45],
           [-50,  30,  13],
           [ 35, -17,  47],
           [-29,  -2, -47],
           [-48, -24,  45],
           [-12, -45, -12],
           [ 17,  -4, -24],
           [ 40, -45, -42],
           [  3,  32,  33],
           [ 36, -40,  -6]])



You can also pass a list of indices for either ``times`` or ``locs`` and
return a subset of the brain object

.. code:: ipython2

    bo_s = bo.get_slice(sample_inds=[1,2,3], loc_inds=[1,2,3])
    bo_s.get_data()




.. parsed-literal::

    array([[ 0.78761269,  0.1053287 ,  0.67986275],
           [ 1.1040404 ,  1.48801596,  1.29897076],
           [-1.49019332, -1.30736317, -1.37526695]])



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
    Date created: Fri Feb 16 13:41:46 2018
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

    <supereeg.brain.Brain at 0x113add150>



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


