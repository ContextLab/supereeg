
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
    0 -16 -38  16
    1 -23 -42  45
    2 -49  -7 -50
    3  43 -49  42
    4 -30  34 -20
    5   2   9 -14
    6  33  10  31
    7  13  24 -50
    8 -38  -9 -43
    9 -32  27  18


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
    Date created: Wed Feb  7 10:40:02 2018
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
    Date created: Wed Feb  7 10:40:03 2018
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
    Date created: Wed Feb  7 10:40:03 2018
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
          <td>-0.772380</td>
          <td>-0.676374</td>
          <td>-0.982353</td>
          <td>-0.464174</td>
          <td>-0.926043</td>
          <td>-0.240970</td>
          <td>-0.212002</td>
          <td>-0.209550</td>
          <td>-1.089664</td>
          <td>-0.358500</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.820856</td>
          <td>-0.163822</td>
          <td>-0.029620</td>
          <td>1.948637</td>
          <td>1.565643</td>
          <td>-0.191169</td>
          <td>2.337410</td>
          <td>1.105355</td>
          <td>0.347662</td>
          <td>0.236310</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.453041</td>
          <td>-0.605926</td>
          <td>-0.131181</td>
          <td>-1.185119</td>
          <td>-0.009260</td>
          <td>-0.469005</td>
          <td>-0.528360</td>
          <td>-0.662305</td>
          <td>-0.273820</td>
          <td>-1.006349</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.302035</td>
          <td>0.076347</td>
          <td>0.445158</td>
          <td>0.777589</td>
          <td>0.597624</td>
          <td>0.170587</td>
          <td>0.822392</td>
          <td>0.396367</td>
          <td>0.284428</td>
          <td>0.343429</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.097325</td>
          <td>-0.174187</td>
          <td>-0.322005</td>
          <td>0.076760</td>
          <td>0.096388</td>
          <td>0.242010</td>
          <td>0.269289</td>
          <td>-0.179111</td>
          <td>-0.177057</td>
          <td>0.436154</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[-0.77238003, -0.67637441, -0.98235279, ..., -0.20954993,
            -1.08966442, -0.3585002 ],
           [-0.8208558 , -0.16382241, -0.0296204 , ...,  1.1053548 ,
             0.34766247,  0.23631027],
           [-0.45304075, -0.60592625, -0.1311814 , ..., -0.66230517,
            -0.27381993, -1.00634944],
           ..., 
           [-0.62848336, -0.43855694, -0.72904693, ..., -0.38210081,
            -0.45892138, -1.26280124],
           [ 0.45914764,  0.18926437, -0.16283932, ..., -0.61136929,
            -0.3427962 , -0.26974233],
           [-0.76976249, -0.54865923, -0.22831248, ..., -0.37224086,
            -0.24674779, -0.44341553]])



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
          <td>-16</td>
          <td>-38</td>
          <td>16</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-23</td>
          <td>-42</td>
          <td>45</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-49</td>
          <td>-7</td>
          <td>-50</td>
        </tr>
        <tr>
          <th>3</th>
          <td>43</td>
          <td>-49</td>
          <td>42</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-30</td>
          <td>34</td>
          <td>-20</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[-16, -38,  16],
           [-23, -42,  45],
           [-49,  -7, -50],
           [ 43, -49,  42],
           [-30,  34, -20],
           [  2,   9, -14],
           [ 33,  10,  31],
           [ 13,  24, -50],
           [-38,  -9, -43],
           [-32,  27,  18]])



You can also plot both the data and the electrode locations:

.. code:: ipython2

    bo.plot_data()
    plt.show()


.. parsed-literal::

    /Users/lucyowen/repos/superEEG/supereeg/brain.py:290: MatplotlibDeprecationWarning: The set_axis_bgcolor function was deprecated in version 2.0. Use set_facecolor instead.
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
    Date created: Wed Feb  7 10:40:03 2018
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

    /Users/lucyowen/repos/superEEG/supereeg/brain.py:388: UserWarning: Voxel sizes of reconstruction and template do not match. Default to using a template with 20mm voxels.
      warnings.warn('Voxel sizes of reconstruction and template do not match. '
    /Users/lucyowen/repos/superEEG/supereeg/brain.py:409: UserWarning: Voxel sizes of reconstruction and template do not match. Voxel sizes calculated from model locations.
      warnings.warn('Voxel sizes of reconstruction and template do not match. '
    /Users/lucyowen/repos/superEEG/supereeg/brain.py:428: RuntimeWarning: invalid value encountered in divide
      data = np.divide(data, counts)

