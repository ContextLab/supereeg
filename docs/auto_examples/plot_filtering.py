# -*- coding: utf-8 -*-
"""
=============================
Filtering electrodes
=============================

This example filters electrodes based on kurtosis thresholding (default=10).

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se

# load example data
bo = se.load('example_filter')

# plot filtered data as default
bo.plot_data()

# plot filtered locations as default
bo.plot_locs()

# 37 locations
bo.info()

# or you can set filter to None if you want to plot original data
bo.filter = None

# plot unfiltered data
bo.plot_data()

# plot unfiltered locations (in aqua)
bo.plot_locs()

# 40 locations
bo.info()
