# Sample Module Repository #

This repository serves as a starting place for all CDL projects.

## To create a new project: ##

1. Create an empty project
2. Clone a copy of this repository
3. Copy the contents of this project into your new (empty) project folder
4. Do an initial commit and push to check this sample code into your new project
5. As you update your code, be sure to also update the following files:
  + Add project dependencies to requirements.txt
  + Update docs/conf.py with your repository name and documentation files
  + Change the name of the "sample" directory to whatever you want your toolbox to be called (add your project code to that new directory, with sub-folders as desired)
  + Update setup.py with your toolbox name and a brief description

## To install your project as a Python package: ##
1. Navigate to the project repository in Terminal
2. Type `pip install -e .`
This should be re-run each time your code is updated and you want your new code to be accessible via your system-wide Python installation.  It assumes that you have [`pip`](https://pip.pypa.io/en/stable/installing/) installed on your computer.

## Instructions to add: ##
+ Instructions for setting up TravisCI tests
+ Instructions for integrating with CDL slack channels
+ Link to lab manual for best coding practices and style guidelines
