# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

DESCRIPTION = 'Infer activity throughout the brain from a small(ish) number of electrodes using Gaussian process regression'
LONG_DESCRIPTION = """\
[supereeg](https://github.com/ContextLab/supereeg>) (name inspired by Robert Sawyer's [The Terminal Experiment]
(https://en.wikipedia.org/wiki/The_Terminal_Experiment) is a (fictional) tool for recording the electrical activities 
of every neuron in the living human brain.  Our approach is somewhat less ambitious, but (we think) still "super" cool: 
obtain high spatiotemporal estimates of activity patterns throughout the brain using data from a small(ish) number of  
[implanted electrodes](https://en.wikipedia.org/wiki/Electrocorticography).  The toolbox is designed to analyze ECoG 
(electrocorticographic) data, e.g. from epilepsy patients undergoing pre-surgical evaluation.

The way the technique works is to leverage data from different patients' brains (who had electrodes implanted in 
different locations) to learn a "correlation model" that describes how activity patterns at different locations 
throughout the brain relate.  Given this model, along with data from a sparse set of locations, we use Gaussian process 
regression to "fill in" what the patients' brains were "most probably" doing when those recordings were taken.  
Details on our approach may be found in [this preprint](http://biorxiv.org/content/early/2017/03/27/121020).  
You may also be interested in watching [this talk](https://youtu.be/DvzfPsOMvOw?t=2s) or reading this [blog post]
(https://community.sfn.org/t/supereeg-ecog-data-breaks-free-from-electrodes/8344) from a recent conference.

Although our toolbox is designed with ECoG data in mind, in theory this tool could be applied to a very general set 
of applications.  The general problem we solve is: given known (correlational) structure of a large number of 
"features," and given that (at any one time) you only observe some of those features, how much can you infer 
about what the remaining features are doing?

Toolbox documentation, including a full API specification, tutorials, and gallery of examples may be 
found [here](http://supereeg.readthedocs.io/) on our readthedocs page.

"""
LICENSE = 'MIT'

setup(
    name='supereeg',
    version='0.1.0',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='Contextual Dynamics Laboratory',
    author_email='contextualdynamics@gmail.com',
    url='https://www.context-lab.com',
    license=LICENSE,
    install_requires=[
        'deepdish',
        'scikit-learn>=0.18.1',
        'pandas>=0.21.1',
        'seaborn>=0.7.1',
        'matplotlib>=2.2.0',
        'scipy>=0.17.1',
        'numpy>=1.10.4',
        'nilearn>=0.4.1',
        'nibabel',
        'joblib',
        'imageio',
        'future',
        'hypertools',
        'six'
    ],
    packages=find_packages(exclude=('tests', 'docs')),
)
