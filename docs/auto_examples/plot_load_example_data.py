# -*- coding: utf-8 -*-
"""
=============================
Loading data
=============================

Here, we load an example dataset and then print out some information about it.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se

# load example data
bo = se.load('example_data')

# check out the brain object (bo)
bo.info()

# look data, stored as pandas dataframe
bo.data.head()

# then can visualize locations
bo.plot_locs()

# visualize the data with plot_data
# the default time window is the first 10 seconds, but you can specify your own timewindow
bo.plot_data(time_min=0, time_max=5)
