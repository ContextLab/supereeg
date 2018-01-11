#!/usr/bin/python

#a simple python job:
# + sleeps for 2 seconds
# + creates a file in the current working directory called <JOBNUM>.results, where <JOBNUM> is the input argument
from time import sleep
import sys
import os
from config import config
import superEEG as se

try:
    os.stat(config['resultsdir'])
except:
    os.makedirs(config['resultsdir'])

gray = se.load('mini_model')
print(gray.locs)
results_file = os.path.join(config['resultsdir'], sys.argv[1]+'.results')

if not os.path.isfile(results_file):
    sleep(2)

    fd = open(results_file, 'w+')
    fd.write(sys.argv[1]+'\n')
    fd.close()

