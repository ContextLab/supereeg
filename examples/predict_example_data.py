# -*- coding: utf-8 -*-
"""
=============================
Predict a dataset
=============================

In this example, we load in a single subject example, remove electrodes that exceed
a kurtosis threshold (in place), load a model, and predict activity at all
model locations.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import superEEG

# load example data
bo = superEEG.load_example_data()

# remove elecs that exceed some threshold
bo.remove_elecs(measure='kurtosis', threshold=10)

# load example model
model = superEEG.load_example_model()

# debug predict.py
p_bo = superEEG.predict(bo, model=model)
