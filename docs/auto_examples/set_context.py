# -*- coding: utf-8 -*-
"""
=============================
Setting a context
=============================

SuperEEG will run in different ways depending your computing environment. To
change the computing context, use the `set_context` function.  You can select
single, cluster or define your own custom computing environment.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

import superEEG as se

# print default context
print(se.context)

# change context to preset cluster default
se.set_context('cluster')

# print updated context
print(se.context)

# change context to custom dict
google = {
    'environment' : 'google-cloud',
    'nodes' : 1000,
    'memory' : 300,
}
se.set_context(google)
print(se.context)
