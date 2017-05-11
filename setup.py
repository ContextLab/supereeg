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
    packages=find_packages(exclude=('tests', 'docs')),
)
