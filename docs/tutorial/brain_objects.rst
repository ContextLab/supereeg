
Building a brain object
=======================

Brain objects are superEEG’s fundamental data structure for a single
subject’s iEEG data. To create one at minimum you’ll need a matrix of
neural recordings (time samples by electrodes), electrode locations, and
a sample rate. Additionally, you can include information about separate
recording sessions and store custom meta data. In this tutorial, we’ll
build a brain object from scratch and get familiar with some of the
methods.

Load in the required libraries
==============================

.. code:: ipython2

    import superEEG as se
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt


.. parsed-literal::

    /Library/Python/2.7/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)


Simulate some data
==================

First, we’ll use superEEG’s built in simulation functions to simulate
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
    0  25  -2  -2
    1  35  25  27
    2  31  20  -8
    3  19 -49 -27
    4  -7  26 -18
    5  44   7   1
    6  29  11 -22
    7 -38 -33  26
    8 -35 -49 -36
    9 -16  32   2


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
    Date created: Mon Feb  5 15:26:50 2018
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
    Date created: Mon Feb  5 15:26:50 2018
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
    Date created: Mon Feb  5 15:26:50 2018
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
          <td>-0.930009</td>
          <td>-0.754930</td>
          <td>0.334511</td>
          <td>0.736630</td>
          <td>0.839893</td>
          <td>-0.203794</td>
          <td>1.252928</td>
          <td>0.543430</td>
          <td>0.202970</td>
          <td>-0.142840</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.663206</td>
          <td>-0.773829</td>
          <td>-1.080991</td>
          <td>-0.631675</td>
          <td>-0.832990</td>
          <td>-0.283372</td>
          <td>-1.071417</td>
          <td>-0.705510</td>
          <td>-1.076166</td>
          <td>-0.776047</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.372385</td>
          <td>0.561954</td>
          <td>0.626500</td>
          <td>-0.353542</td>
          <td>-0.131171</td>
          <td>0.024503</td>
          <td>-0.348883</td>
          <td>-0.229059</td>
          <td>-0.265106</td>
          <td>-0.166613</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-0.496578</td>
          <td>-1.130612</td>
          <td>-0.695002</td>
          <td>-1.301244</td>
          <td>-0.494200</td>
          <td>-0.991676</td>
          <td>-1.423415</td>
          <td>-1.183072</td>
          <td>-0.874190</td>
          <td>-0.960488</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.827513</td>
          <td>0.891721</td>
          <td>0.765209</td>
          <td>0.204045</td>
          <td>0.175431</td>
          <td>0.401281</td>
          <td>-0.394446</td>
          <td>0.383435</td>
          <td>0.648246</td>
          <td>0.780819</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[-0.93000919, -0.75492991,  0.33451071, ...,  0.54342999,
             0.20296994, -0.14284028],
           [-0.6632061 , -0.77382875, -1.08099143, ..., -0.70550965,
            -1.07616636, -0.7760469 ],
           [ 0.37238544,  0.5619545 ,  0.62650049, ..., -0.2290587 ,
            -0.26510573, -0.16661298],
           ..., 
           [ 0.93253886,  0.4405663 ,  0.72032217, ...,  0.42949391,
             0.45702549,  0.10989546],
           [-0.70807758, -1.20442834, -0.91176033, ..., -1.36567471,
            -0.36029023, -0.80815287],
           [ 0.16579536,  0.82929421,  0.32887336, ..., -0.23778818,
            -0.10385272,  0.3207427 ]])



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
          <td>25</td>
          <td>-2</td>
          <td>-2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>35</td>
          <td>25</td>
          <td>27</td>
        </tr>
        <tr>
          <th>2</th>
          <td>31</td>
          <td>20</td>
          <td>-8</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19</td>
          <td>-49</td>
          <td>-27</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-7</td>
          <td>26</td>
          <td>-18</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[ 25,  -2,  -2],
           [ 35,  25,  27],
           [ 31,  20,  -8],
           [ 19, -49, -27],
           [ -7,  26, -18],
           [ 44,   7,   1],
           [ 29,  11, -22],
           [-38, -33,  26],
           [-35, -49, -36],
           [-16,  32,   2]])



You can also plot both the data and the electrode locations:

.. code:: ipython2

    bo.plot_data()
    plt.show()


.. parsed-literal::

    /Users/lucyowen/repos/superEEG/superEEG/brain.py:268: MatplotlibDeprecationWarning: The set_axis_bgcolor function was deprecated in version 2.0. Use set_facecolor instead.
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
    Date created: Mon Feb  5 15:26:50 2018
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

    /Users/lucyowen/repos/superEEG/superEEG/brain.py:366: UserWarning: Voxel sizes of reconstruction and template do not match. Default to using a template with 20mm voxels.
      warnings.warn('Voxel sizes of reconstruction and template do not match. '
    /Users/lucyowen/repos/superEEG/superEEG/brain.py:387: UserWarning: Voxel sizes of reconstruction and template do not match. Voxel sizes calculated from model locations.
      warnings.warn('Voxel sizes of reconstruction and template do not match. '
    /Users/lucyowen/repos/superEEG/superEEG/brain.py:406: RuntimeWarning: invalid value encountered in divide
      data = np.divide(data, counts)

