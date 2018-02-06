
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
    0 -47  27  -1
    1  22  36  28
    2 -32 -45  44
    3 -13 -33  21
    4  30 -23  42
    5 -42 -35  -3
    6 -17  45  26
    7 -41 -43 -21
    8  15 -37  23
    9  25 -33  34


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
    Date created: Tue Feb  6 17:07:48 2018
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
    Date created: Tue Feb  6 17:07:48 2018
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
    Date created: Tue Feb  6 17:07:48 2018
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
          <td>0.272359</td>
          <td>-0.346215</td>
          <td>0.457709</td>
          <td>0.627192</td>
          <td>-0.045052</td>
          <td>0.096945</td>
          <td>0.801392</td>
          <td>0.698281</td>
          <td>0.564309</td>
          <td>0.157804</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.363834</td>
          <td>-0.322107</td>
          <td>-0.269109</td>
          <td>0.948186</td>
          <td>0.900099</td>
          <td>0.053983</td>
          <td>0.730164</td>
          <td>0.727272</td>
          <td>-0.350567</td>
          <td>0.204392</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.876172</td>
          <td>-0.857605</td>
          <td>-0.520450</td>
          <td>-1.284726</td>
          <td>-0.727677</td>
          <td>-0.846195</td>
          <td>-0.724739</td>
          <td>-1.176506</td>
          <td>-0.786837</td>
          <td>-1.075212</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-1.185252</td>
          <td>-1.066109</td>
          <td>-0.916723</td>
          <td>-1.855163</td>
          <td>-1.227675</td>
          <td>-0.843783</td>
          <td>-0.989538</td>
          <td>-1.557971</td>
          <td>-1.778274</td>
          <td>-0.855582</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.726491</td>
          <td>-0.087320</td>
          <td>0.529055</td>
          <td>0.813207</td>
          <td>0.518054</td>
          <td>0.319515</td>
          <td>1.597604</td>
          <td>0.308747</td>
          <td>0.080153</td>
          <td>-0.233562</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[ 0.27235913, -0.34621549,  0.45770948, ...,  0.69828076,
             0.56430904,  0.1578043 ],
           [-0.36383365, -0.32210677, -0.269109  , ...,  0.72727175,
            -0.35056748,  0.2043922 ],
           [-0.87617249, -0.85760538, -0.52044951, ..., -1.17650608,
            -0.78683729, -1.07521218],
           ..., 
           [-0.18642071, -0.78038198, -0.91055963, ..., -0.99044686,
            -0.53508072, -0.66602162],
           [ 0.84686822,  0.63736646,  0.69212157, ..., -0.11737048,
             0.46502722,  1.27989025],
           [-1.5078785 , -2.05440333, -1.0594531 , ..., -0.93034867,
            -0.88913965, -0.82631128]])



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
          <td>-47</td>
          <td>27</td>
          <td>-1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22</td>
          <td>36</td>
          <td>28</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-32</td>
          <td>-45</td>
          <td>44</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-13</td>
          <td>-33</td>
          <td>21</td>
        </tr>
        <tr>
          <th>4</th>
          <td>30</td>
          <td>-23</td>
          <td>42</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[-47,  27,  -1],
           [ 22,  36,  28],
           [-32, -45,  44],
           [-13, -33,  21],
           [ 30, -23,  42],
           [-42, -35,  -3],
           [-17,  45,  26],
           [-41, -43, -21],
           [ 15, -37,  23],
           [ 25, -33,  34]])



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
    Date created: Tue Feb  6 17:07:48 2018
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

