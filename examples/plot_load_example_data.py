# -*- coding: utf-8 -*-
"""
=============================
Loading data
=============================

Here, we load an example dataset and then print out some information about it.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

# import
import superEEG as se
import seaborn as sns
import matplotlib.pyplot as plt

# load example data
bo = se.load('example_data')

# check out the brain object (bo)
bo.info()


# look data, stored as pandas dataframe
bo.data.head()

# and visualize the data

bo.plot()
