# -*- coding: utf-8 -*-
"""
=============================
Filtering electrodes
=============================

This example filters electrodes based on kurtosis thresholding.

"""

# Code source: Lucy Owen & Andrew Heusser
# License: MIT

# import
import supereeg as se

# load example data
bo = se.load('example_filter')

# to get kurtosis values
kurt_vals = bo.kurtosis

# 40 locations before filtering
bo.get_locs()

# plot data will filter by default
bo.plot_data()

# but filtered=False will show all electrodes
bo.plot_data(filtered=False)

# filter elecs, default measure='kurtosis' and threshold=10
f_bo = se.filter_elecs(bo)

# 37 locations after filtering
f_bo.get_locs()
