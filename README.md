![supereeg logo](images/supereeg.png)

<h2>Overview</h2>

[supereeg](https://github.com/ContextLab/supereeg>) (name inspired by Robert Sawyer's [The Terminal Experiment](https://en.wikipedia.org/wiki/The_Terminal_Experiment) is a (fictional) tool for recording the electrical activities of every neuron in the living human brain.  Our approach is somewhat less ambitious, but (we think) still "super" cool: obtain high spatiotemporal estimates of activity patterns throughout the brain using data from a small(ish) number of  [implanted electrodes](https://en.wikipedia.org/wiki/Electrocorticography).  The toolbox is designed to analyze ECoG (electrocorticographic) data, e.g. from epilepsy patients undergoing pre-surgical evaluation.

The way the technique works is to leverage data from different patients' brains (who had electrodes implanted in different locations) to learn a "correlation model" that describes how activity patterns at different locations throughout the brain relate.  Given this model, along with data from a sparse set of locations, we use Gaussian process regression to "fill in" what the patients' brains were "most probably" doing when those recordings were taken.  Details on our approach may be found in [this preprint](http://biorxiv.org/content/early/2017/03/27/121020).  You may also be interested in watching [this talk](https://www.youtube.com/watch?v=t6snLszEneA&feature=youtu.be&t=35) or reading this [blog post](https://community.sfn.org/t/supereeg-ecog-data-breaks-free-from-electrodes/8344) from a recent conference.

Although our toolbox is designed with ECoG data in mind, in theory this tool could be applied to a very general set of applications.  The general problem we solve is: given known (correlational) structure of a large number of "features," and given that (at any one time) you only observe some of those features, how much can you infer about what the remaining features are doing?

Toolbox documentation, including a full API specification, tutorials, and gallery of examples may be found [here](http://supereeg.readthedocs.io/) on our readthedocs page.

<h2>Installation</h2>

<h3>Recommended way of installing the toolbox</h3>

You may install the latest stable version of our toolbox using [pip](https://pypi.org/project/pip/):

`pip install supereeg`

or if you have a previous version already installed:

`pip install --upgrade supereeg`

<h3>Dangerous/hacker/developer way of installing the toolbox (use caution!)</h3>
To install the latest (bleeding edge) version directly from this repository use:

`pip install --upgrade git+https://github.com/ContextLab/supereeg.git`

<h3>One time setup</h3>

1. Install Docker on your computer using the appropriate guide below:
    - [OSX](https://docs.docker.com/docker-for-mac/install/#download-docker-for-mac)
    - [Windows](https://docs.docker.com/docker-for-windows/install/)
    - [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
    - [Debian](https://docs.docker.com/engine/installation/linux/docker-ce/debian/)
2. Launch Docker and adjust the preferences to allocate sufficient resources (e.g. > 4GB RAM)
3. Build the docker image by opening a terminal in the desired folder and enter `docker pull contextualdynamicslab/supereeg`  
4. Use the image to create a new container for the workshop
    - The command below will create a new container that will map your computer's `Desktop` to `/mnt` within the container, so that location is shared between your host OS and the container. Feel free to change `Desktop` to whatever folder you prefer to share instead, but make sure to provide the full path. The command will also share port `8888` with your host computer so any jupyter notebooks launched from *within* the container will be accessible at `localhost:8888` in your web browser (or `192.168.99.100:8888` if using Docker Toolbox)
    - `docker run -it -p 8888:8888 --name supereeg -v ~/Desktop:/mnt contextualdynamicslab/supereeg `
    - You should now see the `root@` prefix in your terminal, if so you've successfully created a container and are running a shell from *inside*!
5. To launch Jupyter: `jupyter notebook --no-browser --ip=0.0.0.0 --allow-root`
6. (Optional) Connect Docker to [PyCharm](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html) or another IDE


<h3>Using the container after setup</h3>

1. You can always fire up the container by typing the following into a terminal
    - `docker start supereeg && docker attach supereeg`
    - When you see the `root@` prefix, letting you know you're inside the container
2. Close a running container with `ctrl + d` from the same terminal you used to launch the container, or `docker stop supereeg` from any other terminal

<h2>Requirements</h2>

The toolbox is currently supported on Mac and Linux.  It has not been tested on Windows (and we expect key functionality not to work properly on Windows systems). If using Windows, consider using Windows Subsystem for Linux or a Docker container.

Dependencies:
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



<h2>Citing</h2>

We wrote a paper about supereeg, which you can read [here](http://biorxiv.org/content/early/2017/03/27/121020).  The paper provides full details about the approach along with some performance tests an a large ECoG dataset.  If you use this toolbox or wish to cite us, please use the following citation:

`Owen LLW and Manning JR (2017) Towards Human Super EEG.  bioRxiv: 121020`

Here is a bibtex formatted reference:

```
@article {Owen121020,
	author = {Owen, Lucy L. W. and Manning, Jeremy R.},
	title = {Towards Human Super EEG},
	year = {2017},
	doi = {10.1101/121020},
	publisher = {Cold Spring Harbor Labs Journals}
	URL = {http://biorxiv.org/content/early/2017/03/27/121020},
	eprint = {http://biorxiv.org/content/early/2017/03/27/121020.full.pdf},
	journal = {bioRxiv}
}

```

<h2>Contributing</h2>

Thanks for considering adding to our toolbox!  Some text below hoas been borrowed from the [Matplotlib contributing guide](http://matplotlib.org/devdocs/devel/contributing.html).

<h3>Submitting a bug report</h3>

If you are reporting a bug, please do your best to include the following:

1. A short, top-level summary of the bug. In most cases, this should be 1-2 sentences.
2. A short, self-contained code snippet to reproduce the bug, ideally allowing a simple copy and paste to reproduce. Please do your best to reduce the code snippet to the minimum required.
3. The actual outcome of the code snippet
4. The expected outcome of the code snippet

<h3>Contributing code</h3>

The preferred way to contribute to supereeg is to fork the main repository on GitHub, then submit a pull request.

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
