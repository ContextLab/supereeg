
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
    0 -49 -25  -4
    1 -30 -31 -27
    2 -37  -3   0
    3   4 -17   1
    4  37  38 -28
    5 -34 -18  49
    6  19  21   7
    7 -27  -3 -38
    8 -23 -44  30
    9   0  -6  23


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
    Date created: Tue Feb  6 14:32:28 2018
    Meta data: None


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
    Date created: Tue Feb  6 14:32:28 2018
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
    Recording time in seconds: [ 0.5  0.5]
    Number of sessions: 2
    Date created: Tue Feb  6 14:32:28 2018
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
          <td>0.101923</td>
          <td>-0.269990</td>
          <td>-0.145735</td>
          <td>-0.735009</td>
          <td>-0.432929</td>
          <td>-0.279040</td>
          <td>-1.118166</td>
          <td>-0.822754</td>
          <td>-0.197954</td>
          <td>-0.383987</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.269369</td>
          <td>-0.591694</td>
          <td>-0.926477</td>
          <td>0.460173</td>
          <td>0.653169</td>
          <td>0.656731</td>
          <td>0.135766</td>
          <td>0.421282</td>
          <td>0.173871</td>
          <td>-0.151593</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.226891</td>
          <td>0.090132</td>
          <td>0.303854</td>
          <td>-0.328004</td>
          <td>0.046752</td>
          <td>0.383989</td>
          <td>0.079332</td>
          <td>0.170870</td>
          <td>0.746654</td>
          <td>0.058811</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.350863</td>
          <td>0.677900</td>
          <td>0.648252</td>
          <td>0.280403</td>
          <td>0.413462</td>
          <td>0.296875</td>
          <td>-0.603022</td>
          <td>0.033148</td>
          <td>0.391847</td>
          <td>0.294850</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.038621</td>
          <td>0.854237</td>
          <td>0.238587</td>
          <td>-0.641726</td>
          <td>-0.271134</td>
          <td>0.731338</td>
          <td>-0.787121</td>
          <td>0.049227</td>
          <td>0.515314</td>
          <td>-0.122781</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[ 0.10192263, -0.26998972, -0.14573523, ..., -0.82275436,
            -0.19795365, -0.38398701],
           [-0.26936855, -0.59169425, -0.92647694, ...,  0.42128186,
             0.1738708 , -0.15159262],
           [ 0.22689076,  0.09013213,  0.30385369, ...,  0.17087022,
             0.74665361,  0.05881138],
           ..., 
           [-0.02022482,  0.87341572,  0.45888157, ..., -0.04092229,
             0.16659371, -0.06983825],
           [ 1.00861557,  1.20858036,  0.55869858, ...,  0.01128124,
            -0.28969225,  0.77505548],
           [-0.03204648, -0.12017914,  0.16919908, ...,  0.2389383 ,
            -0.40173662,  0.5693572 ]])



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
          <td>-49</td>
          <td>-25</td>
          <td>-4</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-30</td>
          <td>-31</td>
          <td>-27</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-37</td>
          <td>-3</td>
          <td>0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>-17</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>37</td>
          <td>38</td>
          <td>-28</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[-49, -25,  -4],
           [-30, -31, -27],
           [-37,  -3,   0],
           [  4, -17,   1],
           [ 37,  38, -28],
           [-34, -18,  49],
           [ 19,  21,   7],
           [-27,  -3, -38],
           [-23, -44,  30],
           [  0,  -6,  23]])



You can also plot both the data and the electrode locations:

.. code:: ipython2

    bo.plot_data()
    plt.show()


.. parsed-literal::

    /Users/lucyowen/repos/superEEG/supereeg/brain.py:268: MatplotlibDeprecationWarning: The set_axis_bgcolor function was deprecated in version 2.0. Use set_facecolor instead.
      ax.set_axis_bgcolor('w')



.. image:: brain_objects_files/brain_objects_23_1.png


.. code:: ipython2

    bo.plot_locs()


.. parsed-literal::

    /Library/Python/2.7/site-packages/matplotlib/cbook.py:136: MatplotlibDeprecationWarning: The axisbg attribute was deprecated in version 2.0. Use facecolor instead.
      warnings.warn(message, mplDeprecation, stacklevel=1)
    /Library/Python/2.7/site-packages/nilearn/plotting/glass_brain.py:164: MatplotlibDeprecationWarning: The get_axis_bgcolor function was deprecated in version 2.0. Use get_facecolor instead.
      black_bg = colors.colorConverter.to_rgba(ax.get_axis_bgcolor()) \
    /Library/Python/2.7/site-packages/nilearn/plotting/displays.py:1259: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if node_color == 'auto':



.. image:: brain_objects_files/brain_objects_24_1.png


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
    Date created: Tue Feb  6 14:32:28 2018
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



.. image:: brain_objects_files/brain_objects_37_0.png


``bo.plot_locs()``
------------------

This method plots electrode locations from brain object:

.. code:: ipython2

    bo.plot_locs()



.. image:: brain_objects_files/brain_objects_39_0.png


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

    /Users/lucyowen/repos/superEEG/supereeg/brain.py:366: UserWarning: Voxel sizes of reconstruction and template do not match. Default to using a template with 20mm voxels.
      warnings.warn('Voxel sizes of reconstruction and template do not match. '
    /Users/lucyowen/repos/superEEG/supereeg/brain.py:387: UserWarning: Voxel sizes of reconstruction and template do not match. Voxel sizes calculated from model locations.
      warnings.warn('Voxel sizes of reconstruction and template do not match. '
    /Users/lucyowen/repos/superEEG/supereeg/brain.py:406: RuntimeWarning: invalid value encountered in divide
      data = np.divide(data, counts)

