
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

    [[ 13  34 -62]
     [-56 -14 -11]
     [ 30  26 -23]
     [-21  70 -12]
     [-57   0 -63]
     [ 47  43 -68]
     [ -9  29  50]
     [-34  59  34]
     [-48 -65 -65]
     [-45  42  25]]


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
    Date created: Wed Sep 13 20:17:43 2017
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
    Date created: Wed Sep 13 20:17:43 2017
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
    Date created: Wed Sep 13 20:17:43 2017
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
          <td>-1.052077</td>
          <td>0.411299</td>
          <td>-0.018917</td>
          <td>1.142291</td>
          <td>-0.772126</td>
          <td>1.581105</td>
          <td>0.681287</td>
          <td>-2.584910</td>
          <td>-0.333072</td>
          <td>-1.042689</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.130488</td>
          <td>-1.571245</td>
          <td>-0.336744</td>
          <td>0.106800</td>
          <td>1.372622</td>
          <td>0.604115</td>
          <td>-0.416115</td>
          <td>1.741251</td>
          <td>1.369756</td>
          <td>-0.384477</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.053217</td>
          <td>-0.958223</td>
          <td>-0.228796</td>
          <td>0.364165</td>
          <td>-1.625921</td>
          <td>-1.269414</td>
          <td>0.610430</td>
          <td>-1.010005</td>
          <td>0.998501</td>
          <td>1.525880</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-0.614946</td>
          <td>-1.290362</td>
          <td>-0.167248</td>
          <td>-0.352178</td>
          <td>1.630558</td>
          <td>0.589432</td>
          <td>1.695709</td>
          <td>0.472131</td>
          <td>-1.141244</td>
          <td>-2.051878</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2.309127</td>
          <td>0.048818</td>
          <td>0.737877</td>
          <td>-1.479824</td>
          <td>2.398797</td>
          <td>0.350753</td>
          <td>-0.693691</td>
          <td>-0.246620</td>
          <td>0.103721</td>
          <td>0.905462</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[-1.05207711,  0.41129918, -0.01891681, ..., -2.58491048,
            -0.33307204, -1.04268895],
           [ 0.13048775, -1.57124465, -0.33674364, ...,  1.74125075,
             1.36975564, -0.38447684],
           [-0.05321725, -0.95822318, -0.22879632, ..., -1.01000469,
             0.9985013 ,  1.52588023],
           ..., 
           [ 0.67137918,  0.89978251, -1.19735951, ...,  0.45099127,
            -0.35413376, -0.8177762 ],
           [ 3.17867666,  1.20029204, -0.76238831, ...,  0.39568591,
            -1.33745293, -0.16721438],
           [-0.6305526 ,  0.26396368, -0.16196349, ..., -0.25395335,
             0.17104155,  0.19482469]])



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
          <td>13</td>
          <td>34</td>
          <td>-62</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-56</td>
          <td>-14</td>
          <td>-11</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30</td>
          <td>26</td>
          <td>-23</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-21</td>
          <td>70</td>
          <td>-12</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-57</td>
          <td>0</td>
          <td>-63</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[ 13,  34, -62],
           [-56, -14, -11],
           [ 30,  26, -23],
           [-21,  70, -12],
           [-57,   0, -63],
           [ 47,  43, -68],
           [ -9,  29,  50],
           [-34,  59,  34],
           [-48, -65, -65],
           [-45,  42,  25]])



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
    Date created: Wed Sep 13 20:17:43 2017
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

