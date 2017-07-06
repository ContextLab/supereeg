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
import superEEG

# load example data
bo = se.load('example_data')

# check out the bo
bo.info()
