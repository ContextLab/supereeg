# -*- coding: utf-8 -*-
"""
=============================
Create a model from scratch, and then update it with new subject data
=============================

In this example, we will simulate a model and update the model with the new data.
First, we'll load in some example locations. Then, we will simulate
correlational structure (a toeplitz matrix) to impose on our simulated data.
We simulate 3 brain objects by sampling 10 locations from example_locs and
create a model from these brain objects. Then, we will simulate an additional
brain object and use the model.update method to update an existing model with
new data.

"""

# Code source: Andrew Heusser & Lucy Owen
# License: MIT

# import libraries
import matplotlib.pyplot as plt
import supereeg as se


# simulate 100 locations
locs = se.simulate_locations(n_elecs=100)

# simulate correlation matrix
R = se.create_cov(cov='toeplitz', n_elecs=len(locs))

# simulate brain objects for the model that subsample n_elecs for each synthetic patient
model_bos = [se.simulate_model_bos(n_samples=1000, sample_rate=1000, locs=locs, sample_locs=10, cov='toeplitz')
             for x in range(3)]

# create the model object
model = se.Model(data=model_bos, locs=locs, n_subs=3)
model.plot_data()

# brain object locations subsetted
sub_locs = locs.sample(10).sort_values(['x', 'y', 'z'])

# simulate a new brain object using the same covariance matrix
bo = se.simulate_bo(n_samples=1000, sample_rate=1000, locs=sub_locs, cov='toeplitz')

# update the model
new_model = model.update(bo, inplace=False)

# initialize subplots
f, (ax1, ax2) = plt.subplots(1, 2)
f.set_size_inches(14,6)

# plot it and set the title
model.plot_data(ax=ax1, show=False, yticklabels=False, xticklabels=False, cbar=True, vmin=0, vmax=1)
ax1.set_title('Before updating model: 3 subjects total')

# plot it and set the title
new_model.plot_data(ax=ax2, show=False, yticklabels=False, xticklabels=False, cbar=True, vmin=0, vmax=1)
ax2.set_title('After updating model: 4 subjects total')

plt.tight_layout()
plt.show()
