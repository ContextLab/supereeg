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
import supereeg as se
import seaborn as sns
import matplotlib.pyplot as plt

# load example data
bo = se.load('example_data')

print(bo.get_slice([0, 100]).info())
