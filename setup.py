# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='superEEG',
    version='0.1.0',
    description='Infer activity throughout the brain from a small(ish) number of electrodes using Gaussian process regression',
    long_description=readme,
    author='Contextual Dynamics Laboratory',
    author_email='contextualdynamics@gmail.com',
    url='https://www.context-lab.com',
    license=license,
	install_requires=[
   'scikit-learn>=0.18.1',
   'pandas',
   'seaborn>=0.7.1',
   'matplotlib>=1.5.1',
   'scipy>=0.17.1',
   'numpy>=1.10.4',
   'nilearn',
   'nibabel',
   'tensorflow',
   ],
    packages=find_packages(exclude=('tests', 'docs')),
)
