

.. _sphx_glr_auto_examples_simulation.py:


=============================
Simulate data
=============================

In this example, we demonstrate the simulate functions.
First, we'll load in some example locations. We simulate
10 brain objects using a subset of locations and the correlational structure
(a toeplitz matrix) to create the model. We then update that model with
one simulated brain object, also create from a subset of locations and the
correlational structure (a toeplitz matrix). Finally, we update the model with
10 more brain objects following the same simulation procedure above.




.. code-block:: python


    # Code source: Andrew Heusser & Lucy Owen
    # License: MIT

    import superEEG as se
    import os
    import sys
    import ast
    import pandas as pd
    import numpy as np
    import pickle
    import seaborn as sns
    from superEEG._helpers.stats import r2z, z2r, corr_column
    import matplotlib.pyplot as plt
    #plt.switch_backend('agg')


    # n_samples
    n_samples = 1000

    # n_electrodes - number of electrodes for reconstructed patient - need to loop over 5:5:130
    n_elecs = range(5, 165, 50)

    # m_patients - number of patients in the model - need to loop over 10:10:50
    m_patients = [5, 10]

    # m_electrodes - number of electrodes for each patient in the model -  25:25:100
    m_elecs = range(5, 165, 50)

    iter_val = 1

    # load nifti to get locations
    gray = se.load('mini_model')

    # extract locations
    gray_locs = gray.locs


    append_d = pd.DataFrame()

    param_grid = [(p,m,n) for p in m_patients for m in m_elecs for n in n_elecs]

    for p, m, n in param_grid:
        d = []

        for i in range(iter_val):


            #create brain objects with m_patients and loop over the number of model locations and subset locations to build model
            model_bos = [se.simulate_model_bos(n_samples=10000, sample_rate=1000, locs=gray_locs, sample_locs = m) for x in range(p)]

            model_locs = pd.DataFrame()
            for i in range(len(model_bos)):
                model_locs = model_locs.append(model_bos[i].locs, ignore_index = True)

            # create model from subsampled gray locations
            model = se.Model(model_bos, locs=gray_locs)

            # brain object locations subsetted entirely from both model and gray locations - for this n > m (this isn't necessarily true, but this ensures overlap)
            sub_locs = gray_locs.sample(n).sort_values(['x', 'y', 'z'])

            # simulate brain object
            bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=gray_locs)

            # parse brain object to create synthetic patient data
            data = bo.data.iloc[:, sub_locs.index]

            # create synthetic patient (will compare remaining activations to predictions)
            bo_sample = se.Brain(data=data.as_matrix(), locs=sub_locs)

            # reconstruct at 'unknown' locations
            recon = model.predict(bo_sample)

            # sample actual data at reconstructed locations
            actual = bo.data.iloc[:, recon.locs.index]

            # correlate reconstruction with actual data
            corr_vals = corr_column(actual.as_matrix(),recon.data.as_matrix())

            # since the numbers of reconstructed locations change, sample the same number to take mean
            corr_vals_sample = np.random.choice(corr_vals, 5)

            d.append({'Numbder of Patients in Model': p, 'Number of Model Locations': m, 'Number of Patient Locations': n, 'Average Correlation': corr_vals_sample.mean(), 'Correlations': corr_vals, 'Model Locations': model_locs.values, 'Patient Locations': bo_sample.locs.values})

        d = pd.DataFrame(d, columns = ['Numbder of Patients in Model', 'Number of Model Locations', 'Number of Patient Locations', 'Average Correlation', 'Correlations', 'Model Locations', 'Patient Locations'])
        append_d = append_d.append(d)
        append_d.index.rename('Iteration', inplace=True)


    new_df=append_d.groupby('Average Correlation').mean()


    if len(np.unique(new_df['Numbder of Patients in Model'])) > 1:

        fig, axs = plt.subplots(ncols=len(np.unique(new_df['Numbder of Patients in Model'])), sharex=True, sharey=True)

        axs_iter = 0
        cbar_ax = fig.add_axes([.92, .3, .03, .4])
        for i in np.unique(new_df['Numbder of Patients in Model']):


            data_plot = append_d[append_d['Numbder of Patients in Model'] == i].pivot_table(index=['Number of Model Locations'], columns='Number of Patient Locations',
                                                                  values='Average Correlation')
            axs[axs_iter].set_title('Patients = '+ str(i))
            sns.heatmap(data_plot, cmap="coolwarm", cbar = axs_iter == 0, ax = axs[axs_iter], cbar_ax = None if axs_iter else cbar_ax)
            axs[axs_iter].invert_yaxis()
            axs_iter+=1

    else:
        for i in np.unique(new_df['Numbder of Patients in Model']):
            data_plot = append_d[append_d['Numbder of Patients in Model'] == i].pivot_table(
                index=['Number of Model Locations'], columns='Number of Patient Locations',
                values='Average Correlation')
            ax = sns.heatmap(data_plot, cmap="coolwarm", vmin=-1, vmax=1)
            ax.invert_yaxis()
            ax.set(xlabel='Number of electrodes from to-be-reconstructed patient', ylabel=' Number of electrodes from patients used to construct model')

    plt.show()
    #plt.savefig('average_correlation_heatmap.pdf')

**Total running time of the script:** ( 0 minutes  0.000 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: simulation.py <simulation.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: simulation.ipynb <simulation.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <http://sphinx-gallery.readthedocs.io>`_
