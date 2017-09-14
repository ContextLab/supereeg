
Model objects and predicting whole brain activity
=================================================

Model objects are superEEG's class that contains the model that
reconstructs full brain activity timeseries from a smaller sample of
electrodes. The superEEG package offers a few premade models that you
can use to reconstruct brain activity, but also a way to create your own
model if you have a dataset of intracranial patient data converted into
the brain object format. This tutorial will go over how to use the
premade models included in this package, as well as make a new model
from scratch.

Load in the required libraries
==============================

.. code:: ipython2

    import superEEG as se
    import seaborn as sns
    from nilearn import plotting
    from nilearn import image

First, let's load in one of the default models. Below is a model that we
made from the pyFR dataset sampled at 20mm resolution. The 'k10' means
that electrodes with a threshold exceeding 10 were removed from the
dataset, and 'r20' means that the model uses a radial basis function of
with a width of 20 mm to 'fill in' nearby electrode sites during the
model creation.

.. code:: ipython2

    model = se.load('pyFR_k10r20_20mm')
    model.info()


.. parsed-literal::

    Number of locations: 170
    Number of subjects: 67
    Date created: Wed Sep 13 20:18:00 2017
    Meta data: None


Visualizing the model
---------------------

The model is comprised of a number of fields. The most important are the
``model.numerator`` and ``model.denominator``. Dividing these two fields
gives a matrix of z-values, where the value in each cell represents the
covariance between every model brain location with every other model
brain location. To view the model, simply call the ``model.plot``
method. This method wraps ``seaborn.heatmap`` to plot the model
(transformed from z to r), so any arguments that ``seaborn.heatmap``
accepts are supported by ``model.plot``.

.. code:: ipython2

    model.plot(xticklabels=False, yticklabels=False)


.. parsed-literal::

    /Users/andyheusser/Documents/github/superEEG/superEEG/model.py:331: RuntimeWarning: invalid value encountered in divide
      sns.heatmap(z2r(np.divide(self.numerator, self.denominator)), **kwargs)



.. image:: model_objects_files/model_objects_7_1.png


Updating the model
------------------

Now, let's say we wanted to update the model with a new subjects data.
To do this, we can use the ``update`` method, passing a new subjects
data as a brain object. First, let's load in an example subjects data:

.. code:: ipython2

    bo = se.load('example_data')
    bo.info()


.. parsed-literal::

    Number of electrodes: 64
    Recording time in seconds: [[ 5.  5.]]
    Number of sessions: 1
    Date created: Wed Sep 13 20:18:00 2017
    Meta data: None


and then update the model:

.. code:: ipython2

    updated_model = model.update(bo)
    updated_model.info()


.. parsed-literal::

    /Users/andyheusser/Documents/github/superEEG/superEEG/_helpers/stats.py:141: RuntimeWarning: divide by zero encountered in log
      return 0.5 * (np.log(1 + r) - np.log(1 - r))
    /Users/andyheusser/Documents/github/superEEG/superEEG/_helpers/stats.py:122: RuntimeWarning: invalid value encountered in true_divide
      return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


.. parsed-literal::

    Number of locations: 233
    Number of subjects: 68
    Date created: Wed Sep 13 20:18:01 2017
    Meta data: None


Note that the model is now comprised of 68 subjects, instead of 67
before we updated it.

Creating a new model
--------------------

In addition to including a few premade models in the ``superEEG``
package, we also provide a way to construct a model from scratch. For
example, if you have a dataset of iEEG patients, we provide a way to
construct a model that will predict whole brain activity. The more
subjects you include in the model, the better it will be! To create a
model, first you'll need to format your subject data into brain objects.
For the purpose of demonstration, we will simulate 10 subjects and
construct the model from that data:

.. code:: ipython2

    n_subs = 10
    bos = [se.simulate_bo(sample_rate=1000) for i in range(n_subs)]
    bos[0].info()


.. parsed-literal::

    Number of electrodes: 10
    Recording time in seconds: 1
    Number of sessions: 1
    Date created: Wed Sep 13 20:18:01 2017
    Meta data: None


As you can see above, each simulated subject has 10 (randomly placed)
'electrodes', with 1 second of data each. To construct a model from
these brain objects, simply pass them to the ``se.Model`` class, and a
new model will be generated:

.. code:: ipython2

    new_model = se.Model(bos)
    new_model.info()


.. parsed-literal::

    /Users/andyheusser/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/numpy/lib/function_base.py:4011: RuntimeWarning: Invalid value encountered in median
      r = func(a, **kwargs)
    /Users/andyheusser/Documents/github/superEEG/superEEG/brain.py:109: UserWarning: No sample rate given.  Setting sample rate to 1000
      warnings.warn('No sample rate given.  Setting sample rate to 1000')


.. parsed-literal::

    Number of locations: 170
    Number of subjects: 10
    Date created: Wed Sep 13 20:18:06 2017
    Meta data: None


By default, the model is in MNI coordinates with 20mm resolution, but
can easily be switched to a different space using some templates we
include in the package (Xmm, 8mm, 20mm), or your own custom space (note:
the model space MUST be in MNI coordinates).

.. code:: ipython2

    # new_model = se.Model(bos, template='/your/custom/MNI_template.nii')
    # new_model.info()

You can also pass a list (or numpy array) of custom MNI locations to
predict:

.. code:: ipython2

    new_model = se.Model(bos, locs=[[0,0,0],[0,0,1]])
    new_model.info()


.. parsed-literal::

    Number of locations: 2
    Number of subjects: 10
    Date created: Wed Sep 13 20:18:06 2017
    Meta data: None


Predicting whole brain activity
-------------------------------

Now for the magic. ``superEEG`` uses ***gaussian process regression***
to infer whole brain activity given a smaller sampling of electrode
recordings. To predict activity, simply call the ``predict`` method of a
model and pass the subjects brain activity that you'd like to
reconstruct:

.. code:: ipython2

    # plot a slice of the original data
    print('BEFORE')
    print('------')
    bo.info()
    nii = bo.to_nii()
    nii_0 = image.index_img(nii, 1)
    plotting.plot_glass_brain(nii_0)
    plotting.show()
    
    # voodoo magic
    bor = model.predict(bo)
    
    # plot a slice of the whole brain data
    print('AFTER')
    print('------')
    bor.info()
    nii = bor.to_nii()
    nii_0 = image.index_img(nii, 1)
    plotting.plot_glass_brain(nii_0)
    plotting.show()


.. parsed-literal::

    BEFORE
    ------
    Number of electrodes: 64
    Recording time in seconds: [[ 5.  5.]]
    Number of sessions: 1
    Date created: Wed Sep 13 20:18:00 2017
    Meta data: None



.. image:: model_objects_files/model_objects_23_1.png


.. parsed-literal::

    /Users/andyheusser/Documents/github/superEEG/superEEG/model.py:201: RuntimeWarning: invalid value encountered in divide
      model_corrmat_x = np.divide(np.nansum(np.dstack((self.numerator, num_corrmat_x)), 2), self.denominator + denom_corrmat_x)
    /Users/andyheusser/Documents/github/superEEG/superEEG/model.py:227: RuntimeWarning: invalid value encountered in divide
      model_corrmat_x = np.divide(num_corrmat_x, denom_corrmat_x)


.. parsed-literal::

    AFTER
    ------
    Number of electrodes: 170
    Recording time in seconds: [[ 5.  5.]]
    Number of sessions: 1
    Date created: Wed Sep 13 20:18:21 2017
    Meta data: None



.. image:: model_objects_files/model_objects_23_4.png


Using the ``superEEG`` algorithm, we've 'reconstructed' whole brain
activity from a smaller sample of electrodes.
