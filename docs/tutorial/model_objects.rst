
Model objects and predicting whole brain activity
=================================================

Model objects are supereeg’s class that contains the model that
reconstructs full brain activity timeseries from a smaller sample of
electrodes. The supereeg package offers a few premade models that you
can use to reconstruct brain activity, but also a way to create your own
model if you have a dataset of intracranial patient data converted into
the brain object format. This tutorial will go over how to use the
premade models included in this package, as well as make a new model
from scratch.

Load in the required libraries
==============================

.. code:: ipython2

    import warnings 
    warnings.simplefilter("ignore")
    %matplotlib inline
    import supereeg as se
    import numpy as np

First, let’s load in our default model, ``example_model``, that we made
from the pyFR dataset sampled at 20mm resolution. Electrodes with a
threshold exceeding 10 were removed from the dataset, and the radial
basis function of with a width of 20 mm was used to ‘fill in’ nearby
electrode sites during the model creation.

.. code:: ipython2

    model = se.load('example_model')

other model options:

``pyFR_k10r20_6mm``

``pyFR_k10r20_20mm``

Note: The last option is the same as the example_model, but saved as
``.npz`` instead of ``.mo``.

Initialize model objects
========================

Model objects can be initialized by passing a model object (ending in
``.mo``), but can also be initialized with a brain object or nifti
object by specifying ``return_type`` as ``mo`` in the load function.

For example, you can load a nifti object as a model object:

.. code:: ipython2

    se.load('example_nift', return_type='mo')




.. parsed-literal::

    <supereeg.model.Model at 0x1107b38d0>



Model object methods
====================

There are a few other useful methods on a model object:

``mo.info()``
-------------

This method will give you a summary of the model object:

.. code:: ipython2

    model.info()


.. parsed-literal::

    Number of locations: 210
    Number of subjects: 67
    Date created: Thu Mar  8 10:17:39 2018
    Meta data: None


``mo.plot_data()``
------------------

This method will plot your model.

The model is comprised of a number of fields. The most important are the
``model.numerator`` and ``model.denominator``. Dividing these two fields
gives a matrix of z-values, where the value in each cell represents the
covariance between every model brain location with every other model
brain location. To view the model, simply call the ``model.plot``
method. This method wraps ``seaborn.heatmap`` to plot the model
(transformed from z to r), so any arguments that ``seaborn.heatmap``
accepts are supported by ``model.plot``.

.. code:: ipython2

    model.plot_data(xticklabels=False, yticklabels=False)



.. image:: model_objects_files/model_objects_12_0.png




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1178d3550>



``mo.plot_locs()``
------------------

This method will plot the locations in your model.

.. code:: ipython2

    model.plot_locs()



.. image:: model_objects_files/model_objects_14_0.png


``mo.update()``
---------------

This method allows you to update the model with addition subject data.

To do this, we can use the ``update`` method, passing a new subjects
data as a brain object. First, let’s load in an example subjects data:

.. code:: ipython2

    bo = se.load('example_data')
    bo.info()


.. parsed-literal::

    Number of electrodes: 64
    Recording time in seconds: [[ 5.3984375 14.1328125]]
    Sample Rate in Hz: [256, 256]
    Number of sessions: 2
    Date created: Thu Mar  8 10:31:48 2018
    Meta data: {'patient': 'CH003'}


Now you can update the model with that brain object:

.. code:: ipython2

    updated_model = model.update(bo)
    updated_model.info()


.. parsed-literal::

    Number of locations: 210
    Number of subjects: 68
    Date created: Thu Mar  8 10:31:49 2018
    Meta data: None


Note that the model is now comprised of 67 subjects, instead of 66
before we updated it.

``mo.save(fname='something')``
------------------------------

This method will save the brain object to the specified file location.
The data will be saved as a ‘bo’ file, which is a dictionary containing
the elements of a brain object saved in the hd5 format using
``deepdish``.

.. code:: ipython2

    #mo.save(fname='model_object')

Creating a new model
--------------------

In addition to including a few premade models in the ``supereeg``
package, we also provide a way to construct a model from scratch.

Created from a list of brain objects:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, if you have a dataset of iEEG patients, we provide a way to
construct a model that will predict whole brain activity. The more
subjects you include in the model, the better it will be! To create a
model, first you’ll need to format your subject data into brain objects.
For the purpose of demonstration, we will simulate 100 locations across
10 subjects and construct the model from that data:

.. code:: ipython2

    # simulate 100 locations
    locs = se.simulate_locations(100)
    
    # simulate 10 brain objects to create a model
    n_subs = 10
    model_bos = [se.simulate_model_bos(n_samples=1000, sample_rate=1000, sample_locs=20, 
                                       locs=locs, cov='toeplitz') for x in range(n_subs)]
    
    model_bos[0].info()


.. parsed-literal::

    Number of electrodes: 20
    Recording time in seconds: [1.]
    Sample Rate in Hz: [1000]
    Number of sessions: 1
    Date created: Thu Mar  8 10:31:49 2018
    Meta data: {}


As you can see above, each simulated subject has 10 (randomly placed)
‘electrodes’, with 1 second of data each. To construct a model from
these brain objects, simply pass them to the ``se.Model`` class, and a
new model will be generated:

.. code:: ipython2

    new_model = se.Model(data=model_bos, locs=locs)
    new_model.info()


.. parsed-literal::

    Number of locations: 100
    Number of subjects: 10
    Date created: Thu Mar  8 10:31:52 2018
    Meta data: None


Created by adding to model object fields:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another option is to add a model directly.

You can add your model to ``model.numerator``, which comprises the sum
of the zscored correlation matrices over subjects. The
``model.denominator`` field comprises the sum of the number of subjects
contributing to each matrix cell in the ``model.numerator`` field. You
can add the locations for the model in the field ``locs`` and the number
of subjects to ``n_subs``.

In this next example, we’re constructing the model from a toeplitz
matrix with 10 subjects. We’ve updated the ``model.numerator`` field
with the toeplitz matrix created in the function, ``se.create_cov``, and
locations loaded from ``example_locations``. The ``model.denominator``
field in this case is a matrix of ones, but should be the number of
subjects that contributed to each cell in the ``model.numerator`` field.

You can also create a custom covariance matrix in ``se.create_cov`` by
simply passing numpy array as and that is number of locations by number
of locations to ``cov`` and the number of location to ``n_elecs``.

.. code:: ipython2

    R = se.create_cov(cov='toeplitz', n_elecs=len(locs))
    p = 10
    toe_model = se.Model(numerator=R, denominator=np.ones(np.shape(R)), locs=locs, n_subs=p)
    toe_model.plot_data(xticklabels=False, yticklabels=False)



.. image:: model_objects_files/model_objects_27_0.png




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1184d3a10>



In this example we passed a numpy array of custom MNI locations to
predict.

However coordinates can also be derived by specifiying a ``template``
nifti file. By default, the model is in MNI coordinates derived from a
gray matter masked brain at 6mm resolution, but this can be easily
switched to a different space specifying either the standard brain
(``std``) or gray matter masked brain (``gray``) as ``template`` as well
as desired resolution passed as ``vox_size`` , or your own custom space
(note: the model space MUST be in MNI coordinates).

.. code:: ipython2

    # new_model = se.Model(bos, template='/your/custom/MNI_template.nii')

Predicting whole brain activity
-------------------------------

``mo.predict()``
----------------

Now for the magic. ``supereeg`` uses ***gaussian process regression***
to infer whole brain activity given a smaller sampling of electrode
recordings. To predict activity, simply call the ``predict`` method of a
model and pass the subjects brain activity that you’d like to
reconstruct:

``mo.predict(nearest_neighbor=True)``
-------------------------------------

As default, the nearest voxel for each subject’s electrode location is
found and used as revised electrodes location matrix in the prediction.
If ``nearest_neighbor`` is set to ``False``, the original locations are
used in the prediction.

``mo.predict(force_update=False)``
----------------------------------

As default, the model is not updated with the subject’s correlation
matrix. By setting ``force_update`` to ``True``, you will update the
model with the subject’s correlation matrix.

.. code:: ipython2

    # plot a slice of the original data
    print('BEFORE')
    print('------')
    bo.info()
    nii = bo.to_nii(template='gray', vox_size=20)
    nii.plot_glass_brain()
    
    # voodoo magic
    bor = model.predict(bo)
    
    
    # plot a slice of the whole brain data
    print('AFTER')
    print('------')
    bor.info()
    nii = bor.to_nii(template='gray', vox_size=20)
    nii.plot_glass_brain()


.. parsed-literal::

    BEFORE
    ------
    Number of electrodes: 64
    Recording time in seconds: [[ 5.3984375 14.1328125]]
    Sample Rate in Hz: [256, 256]
    Number of sessions: 2
    Date created: Thu Mar  8 10:31:48 2018
    Meta data: {'patient': 'CH003'}



.. image:: model_objects_files/model_objects_33_1.png


.. parsed-literal::

    AFTER
    ------
    Number of electrodes: 210
    Recording time in seconds: [ 5.3984375 14.1328125]
    Sample Rate in Hz: [256, 256]
    Number of sessions: 2
    Date created: Thu Mar  8 10:31:54 2018
    Meta data: {}



.. image:: model_objects_files/model_objects_33_3.png


Using the ``supereeg`` algorithm, we’ve ‘reconstructed’ whole brain
activity from a smaller sample of electrodes.

.. code:: ipython2

    ## this doesn't look great
    
    nii.plot_anat()



.. image:: model_objects_files/model_objects_35_0.png


