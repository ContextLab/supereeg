![supereeg logo](images/supereeg.png)

<h2>Overview</h2>

supereeg is a Python package used for estimating neural activity throughout the brain from a small number of intracranial recordings by leveraging across patients and Gaussian process regression.

<h2>Details</h2>

A fundamental trade-off in human neuroimaging pertains to balancing spatial versus temporal resolution.  For example, techniques like fMRI yield high spatial resolution brain recordings (but at poor temporal resolutions), whereas EEG yields high temporal resolution brain recordings (but at poor spatial resolutions).  Other recording methods, such as from implanted electrodes, yield good spatial and temporal resolutions, but have poor coverage.
supereeg , which is inspired by Gaussian process regression, uses ongoing recordings from implanted electrodes to ‘fill in’ high spatiotemporal resolution activity patterns throughout the brain even at unrecorded locations.
By using a large dataset comprised of hundreds of hours of data collected from 67 neurosurgical patients, we have fit a detailed covariance model describing relations between the activity patterns at different locations throughout the brain.  Collectively the dataset contains recordings from a broad range of locations throughout the brain, but each individual patient contributes only a limited set of locations.  Given this covariance model, and given the recordings at a limited set of locations from an individual patient, supereeg estimates what the recordings would have looked like for that patient if the electrodes had instead been implanted elsewhere throughout their brain.  The result is a detailed high spatiotemporal resolution estimate of evolving activity patterns throughout the brain, at fMRI-like spatial resolutions and EEG-like temporal resolutions.

<!-- <h2>Try it!</h2>

Click the badge to launch a binder instance with example uses:

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/contextlab/quail-example-notebooks)

or

Check the [repo](https://github.com/ContextLab/quail-example-notebooks) of Jupyter notebooks. -->

<h2>Installation</h2>

`pip install supereeg`

or

To install from this repo:

`git clone https://github.com/ContextLab/supereeg.git`

Then, navigate to the folder and type:

`pip install -e .`

(this assumes you have [pip](https://pip.pypa.io/en/stable/installing/) installed on your system)

<h2>Requirements</h2>

The toolbox is currently supported on Mac/Linux (but not Windows).

+ python 2.7, 3.5+
+ pandas>=0.21.1
+ seaborn>=0.7.1
+ matplotlib==2.1.0
+ scipy>=0.17.1
+ numpy>=1.10.4
+ scikit-learn>=0.18.1
+ nilearn
+ nibabel
+ joblib
+ multiprocessing
+ deepdish
+ future
+ imageio
+ hypertools
+ scikit-image
+ pytest (for development)


If installing from github (instead of pip), you must also install the requirements:
`pip install -r requirements.txt`

<h2>Documentation</h2>

Check out our readthedocs: [here](http://supereeg.readthedocs.io/en/latest/).

<h2>Citing</h2>

We wrote a paper about supereeg, which you can read [here](http://biorxiv.org/content/early/2017/03/27/121020).
Please cite as:

`Owen LLW and Manning JR (2017) Towards Human Super EEG.  bioRxiv: 121020`

Here is a bibtex formatted reference:

```
@article {Owen121020,
	author = {Owen, Lucy L. W. and Manning, Jeremy R.},
	title = {Towards Human Super EEG},
	year = {2017},
	doi = {10.1101/121020},
	publisher = {Cold Spring Harbor Labs Journals},
	abstract = {Human Super EEG entails measuring ongoing activity from every cell in a living human brain at millisecond-scale temporal resolutions. Although direct cell-by-cell Super EEG recordings are impossible using existing methods, here we present a technique for inferring neural activity at arbitrarily high spatial resolutions using human intracranial electrophysiological recordings. Our approach, based on Gaussian process regression, relies on two assumptions. First, we assume that some of the correlational structure of people{\textquoteright}s brain activity is similar across individuals. Second, we resolve ambiguities in the data by assuming that neural activity from nearby sources will tend to be similar, all else being equal. One can then ask, for an arbitrary individual{\textquoteright}s brain: given what we know about the correlational structure of other people{\textquoteright}s brains, and given the recordings we made from electrodes implanted in this person{\textquoteright}s brain, how would those recordings most likely have looked at other locations throughout this person{\textquoteright}s brain?},
	URL = {http://biorxiv.org/content/early/2017/03/27/121020},
	eprint = {http://biorxiv.org/content/early/2017/03/27/121020.full.pdf},
	journal = {bioRxiv}
}

```

<h2>Contributing</h2>

(Some text borrowed from Matplotlib contributing [guide](http://matplotlib.org/devdocs/devel/contributing.html).)

<h3>Submitting a bug report</h3>

If you are reporting a bug, please do your best to include the following:

1. A short, top-level summary of the bug. In most cases, this should be 1-2 sentences.
2. A short, self-contained code snippet to reproduce the bug, ideally allowing a simple copy and paste to reproduce. Please do your best to reduce the code snippet to the minimum required.
3. The actual outcome of the code snippet
4. The expected outcome of the code snippet

<h3>Contributing code</h3>

The preferred way to contribute to quail is to fork the main repository on GitHub, then submit a pull request.

+ If your pull request addresses an issue, please use the title to describe the issue and mention the issue number in the pull request description to ensure a link is created to the original issue.

+ All public methods should be documented in the README.

+ Each high-level plotting function should have a simple example in the examples folder. This should be as simple as possible to demonstrate the method.

+ Changes (both new features and bugfixes) should be tested using `pytest`.  Add tests for your new feature to the `tests/` repo folder.

+ Please note that the code is currently in beta thus the API may change at any time. BE WARNED.

<h2>Testing</h2>

<!-- [![Build Status](https://travis-ci.com/ContextLab/quail.svg?token=hxjzzuVkr2GZrDkPGN5n&branch=master) -->

To test supereeg, install pytest (`pip install pytest`) and run `pytest` in the supereeg folder

<!-- <h2>Examples</h2> -->

<!-- See [here](http://cdl-quail.readthedocs.io/en/latest/auto_examples/index.html) for more examples. -->
