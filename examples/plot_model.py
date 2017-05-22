# -*- coding: utf-8 -*-
"""
=============================
Load and plot a model
=============================

Here we load the example model, and then plot it.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

# import
import superEEG

# load the model
model = superEEG.load_example_model()

# plot it
model.plot()
