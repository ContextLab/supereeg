# -*- coding: utf-8 -*-
"""
=============================
Predict unknown location
=============================

In this example, we load in a single subject example, remove electrodes that exceed
a kurtosis threshold (in place), load a model, and predict activity at all
model locations.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import superEEG as se
import seaborn as sns
import numpy as np

# load example data
bo = se.load('example_data')

# save as nifti
new_locs = bo.to_nifti('test')

sns.heatmap(np.divide(bo.locs, new_locs))
sns.plt.show()
