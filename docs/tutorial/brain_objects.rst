
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
electrodes with ``n_elecs``:

.. code:: ipython2

    # simulate some data
    data = se.simulate_bo(n_samples=1000, sessions=2, n_elecs=10).get_data()
    
    # plot it
    plt.plot(data)
    plt.xlabel('time samples')
    plt.ylabel('activation')
    plt.show()



.. image:: brain_objects_files/brain_objects_4_0.png


.. code:: ipython2

    # you can also specify a random seed and set the noise parameter to 0 
    # if you want to simulate the same brain object again
    
    data1 = se.simulate_bo(n_samples=1000, sessions=2, n_elecs=5, random_seed=True, noise=0).get_data()
    data2 = se.simulate_bo(n_samples=1000, sessions=2, n_elecs=5, random_seed=True, noise=0).get_data()
    np.allclose(data1, data2)




.. parsed-literal::

    True



We’ll also simulate some electrode locations

.. code:: ipython2

    locs = se.simulate_locations()
    print(locs)


.. parsed-literal::

        x   y   z
    0 -26  37  17
    1   2 -39 -34
    2  19  49 -27
    3 -11 -13 -16
    4 -21  22   7
    5  45 -47 -35
    6 -48 -47 -28
    7  35 -37  33
    8 -40   6  42
    9 -10 -11  -6


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
    Date created: Fri Feb  2 10:21:05 2018
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
    Date created: Fri Feb  2 10:21:05 2018
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
    Date created: Fri Feb  2 10:21:05 2018
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
          <td>1.402425</td>
          <td>1.507506</td>
          <td>1.509823</td>
          <td>0.798233</td>
          <td>0.309975</td>
          <td>1.000785</td>
          <td>0.714815</td>
          <td>0.925150</td>
          <td>0.940090</td>
          <td>0.665643</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.393498</td>
          <td>0.798619</td>
          <td>0.352916</td>
          <td>-0.501393</td>
          <td>-0.112511</td>
          <td>0.299668</td>
          <td>-0.345194</td>
          <td>-0.185758</td>
          <td>-0.008669</td>
          <td>0.486264</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.018856</td>
          <td>0.098373</td>
          <td>-0.061930</td>
          <td>0.505044</td>
          <td>0.062024</td>
          <td>0.395689</td>
          <td>0.401702</td>
          <td>0.219433</td>
          <td>0.496201</td>
          <td>0.319220</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.419532</td>
          <td>0.535559</td>
          <td>1.173978</td>
          <td>-0.131416</td>
          <td>1.077796</td>
          <td>0.188323</td>
          <td>0.358555</td>
          <td>0.168001</td>
          <td>0.343332</td>
          <td>0.418130</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-0.905368</td>
          <td>-1.239255</td>
          <td>-1.538443</td>
          <td>-0.108556</td>
          <td>-0.048805</td>
          <td>-0.696387</td>
          <td>-0.184118</td>
          <td>-0.285191</td>
          <td>-0.841160</td>
          <td>-1.698069</td>
        </tr>
      </tbody>
    </table>
    </div>



or returned as a numpy array using the ``get_data`` method:

.. code:: ipython2

    bo.get_data()




.. parsed-literal::

    array([[ 1.4024247 ,  1.50750589,  1.50982256, ...,  0.92515009,
             0.94008997,  0.66564312],
           [ 0.39349802,  0.79861949,  0.35291604, ..., -0.18575847,
            -0.00866945,  0.48626406],
           [ 0.01885617,  0.09837292, -0.06193032, ...,  0.21943324,
             0.49620131,  0.31922024],
           ..., 
           [ 0.72083748,  0.20572967,  0.30822715, ...,  0.94712354,
             0.41900735,  0.75787876],
           [-1.9247862 , -1.02016648, -1.25933994, ..., -0.02487984,
            -0.96613633, -1.18037104],
           [ 0.75922625,  0.89205645,  0.75038871, ...,  0.15272148,
             0.59340917,  0.10053925]])



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
          <td>-26</td>
          <td>37</td>
          <td>17</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>-39</td>
          <td>-34</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19</td>
          <td>49</td>
          <td>-27</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-11</td>
          <td>-13</td>
          <td>-16</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-21</td>
          <td>22</td>
          <td>7</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython2

    bo.get_locs()




.. parsed-literal::

    array([[-26,  37,  17],
           [  2, -39, -34],
           [ 19,  49, -27],
           [-11, -13, -16],
           [-21,  22,   7],
           [ 45, -47, -35],
           [-48, -47, -28],
           [ 35, -37,  33],
           [-40,   6,  42],
           [-10, -11,  -6]])



You can also plot both the data and the electrode locations:

.. code:: ipython2

    bo.plot_data()
    plt.show()



.. image:: brain_objects_files/brain_objects_24_0.png


.. code:: ipython2

    bo.plot_locs()


.. parsed-literal::

    /Library/Python/2.7/site-packages/nilearn/plotting/displays.py:1259: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if node_color == 'auto':



.. image:: brain_objects_files/brain_objects_25_1.png


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
    Date created: Fri Feb  2 10:21:05 2018
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

``bo.plot_locs()``
------------------

This method plots electrode locations from brain object:

.. code:: ipython2

    bo.plot_locs()



.. image:: brain_objects_files/brain_objects_40_0.png



.. image:: brain_objects_files/brain_objects_40_1.png


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

    /Users/lucyowen/repos/superEEG/superEEG/brain.py:397: RuntimeWarning: invalid value encountered in divide
      data = np.divide(data, counts)


