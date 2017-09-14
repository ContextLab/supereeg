
Building a brain object
=======================

Brain objects are superEEG's fundamental data structure for a single
subject's iEEG data. To create one at minimum you'll need a matrix of
neural recordings (time samples by electrodes), electrode locations, and
a sample rate. Additionally, you can include information about separate
recording sessions and store custom meta data. In this tutorial, we'll
build a brain object from scratch and get familiar with some of the
methods.

Load in the required libraries
==============================

.. code:: ipython2

    import superEEG as se
    import numpy as np
    import seaborn as sns

Simulate some data
==================

First, we'll use superEEG's built in simulation functions to simulate
some data and electrodes. By default, the ``simualate_data`` function
will return a 1000 samples by 10 electrodes matrix, but you can specify
the number of time samples with ``n_samples`` and the number of
electrodes with ``n_elecs``:

.. code:: ipython2

    # simulate some data
    data = se.simulate_data(n_samples=10000, n_elecs=10)
    
    # plot it
    sns.plt.plot(data)
    sns.plt.xlabel('time samples')
    sns.plt.ylabel('activation')
    sns.plt.show()



.. image:: brain_objects_files/brain_objects_4_0.png


We'll also simulate some electrode locations

.. code:: ipython2

    locs = se.simulate_locations()
    print(locs)


.. parsed-literal::

    [[ 20  43  73]
     [-64  70  -6]
     [-57 -24 -39]
     [-34   5 -76]
     [-76 -20   4]
     [-26   0 -53]
     [-58 -50 -15]
     [ 67  40  65]
     [-33  51  67]
     [-17 -61 -26]]


Creating a brain object
=======================

To construct a new brain objects, simply pass the data and locations to
the ``Brain`` class like this:

.. code:: ipython2

    bo = se.Brain(data=data, locs=locs, sample_rate=1000)

To view a summary of the contents of the brain object, you can call the
``info`` function:

.. code:: ipython2

    bo.info()


.. parsed-literal::

    Number of electrodes: 10
    Recording time in seconds: 10
    Number of sessions: 1
    Date created: Wed Sep 13 20:44:14 2017
    Meta data: None


Optionally, you can pass a ``sessions`` parameter, which is a numpy
array the length of your data with a unique identifier for each session.
For example:

.. code:: ipython2

    sessions = np.array([1]*(data.shape[0]/2)+[2]*(data.shape[0]/2))
    bo = se.Brain(data=data, locs=locs, sample_rate=1000, sessions=sessions)
    bo.info()


.. parsed-literal::

    Number of electrodes: 10
    Recording time in seconds: 10
    Number of sessions: 2
    Date created: Wed Sep 13 20:44:14 2017
    Meta data: None


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
    Recording time in seconds: 10
    Number of sessions: 2
    Date created: Wed Sep 13 20:44:14 2017
    Meta data: {'Hospital': 'DHMC', 'subjectID': '123', 'Investigator': 'Andy'}


The structure of a brain object
===============================

Inside the brain object, the iEEG data is stored as a Pandas DataFrame
that can be accessed directly:

.. code:: ipython2

    bo.data.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
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
          <td>0.736689</td>
          <td>-1.351790</td>
          <td>-0.354450</td>
          <td>0.097813</td>
          <td>1.738638</td>
          <td>-0.191214</td>
          <td>0.430361</td>
          <td>-1.533757</td>
          <td>-2.669739</td>
          <td>1.964575</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.782859</td>
          <td>-0.485609</td>
          <td>0.774961</td>
          <td>-0.044896</td>
          <td>-0.066135</td>
          <td>1.345479</td>
          <td>-1.780028</td>
          <td>-1.552078</td>
          <td>0.493869</td>
          <td>1.244928</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-1.477205</td>
          <td>0.887424</td>
          <td>-0.644102</td>
          <td>-1.692734</td>
          <td>0.478760</td>
          <td>0.410642</td>
          <td>1.296110</td>
          <td>0.786275</td>
          <td>0.283199</td>
          <td>-0.621038</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.460382</td>
          <td>0.572336</td>
          <td>1.025309</td>
          <td>0.581930</td>
          <td>-2.071887</td>
          <td>-0.420389</td>
          <td>1.007538</td>
          <td>-0.441497</td>
          <td>1.121191</td>
          <td>0.050903</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.105034</td>
          <td>-2.549072</td>
          <td>-0.363674</td>
          <td>-0.613376</td>
          <td>-1.379841</td>
          <td>-0.473291</td>
          <td>0.266435</td>
          <td>-0.044713</td>
          <td>-0.748226</td>
          <td>0.823629</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[ 0.73668947, -1.35179023, -0.35445011, ..., -1.53375656,
            -2.66973883,  1.96457473],
           [-0.78285937, -0.48560937,  0.77496069, ..., -1.55207771,
             0.49386939,  1.24492799],
           [-1.47720453,  0.88742391, -0.64410235, ...,  0.78627512,
             0.28319937, -0.621038  ],
           ..., 
           [-0.04787538,  0.72265132,  0.75719168, ...,  1.67286872,
            -1.21309623,  1.40871669],
           [ 0.15541992, -0.44662719, -0.21052171, ..., -0.01532474,
             1.53284149,  0.33198072],
           [ 0.61143495, -1.16511284,  1.20299687, ...,  1.58987572,
            -1.89189214,  0.39488465]])



Similarly, the electrode locations are stored as a Pandas DataFrame, and
can be retrieved as a numpy array using the ``get_locs`` method:

.. code:: ipython2

    bo.locs.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
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
          <td>20</td>
          <td>43</td>
          <td>73</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-64</td>
          <td>70</td>
          <td>-6</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-57</td>
          <td>-24</td>
          <td>-39</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-34</td>
          <td>5</td>
          <td>-76</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-76</td>
          <td>-20</td>
          <td>4</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[ 20,  43,  73],
           [-64,  70,  -6],
           [-57, -24, -39],
           [-34,   5, -76],
           [-76, -20,   4],
           [-26,   0, -53],
           [-58, -50, -15],
           [ 67,  40,  65],
           [-33,  51,  67],
           [-17, -61, -26]])



The other peices of the brain object are listed below:

.. code:: ipython2

    # array of session identifiers for each timepoint
    sessions = bo.sessions
    
    # number of sessions
    n_sessions = bo.n_sessions
    
    # sample rate
    sample_rate = bo.sample_rate
    
    # number of electrodes
    n_elecs = bo.n_elecs
    
    # length of recording in seconds
    n_seconds = bo.n_secs
    
    # the date and time that the bo was created
    date_created = bo.date_created
    
    # kurtosis of each electrode
    kurtosis = bo.kurtosis
    
    # meta data
    meta = bo.meta

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
    Recording time in seconds: 10
    Number of sessions: 2
    Date created: Wed Sep 13 20:44:14 2017
    Meta data: {'Hospital': 'DHMC', 'subjectID': '123', 'Investigator': 'Andy'}


``bo.get_data()``
-----------------

This method will return a numpy array of the data:

.. code:: ipython2

    data_array = bo.get_data()

``bo.get_locs()``
-----------------

This method will return a numpy array of the electrode locations:

.. code:: ipython2

    locs = bo.get_locs()

``bo.save('filepath')``
-----------------------

This method will save the brain object to the specified file location:

.. code:: ipython2

    bo.save('brain_object')


.. parsed-literal::

    Brain object saved as pickle.


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

