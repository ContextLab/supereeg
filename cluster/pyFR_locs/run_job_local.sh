#!/bin/bash -l

# DO NOT MODIFY THIS FILE!

# set the working directory *of the job* to the specified start directory
cd <config['startdir']>

# run the job
<config['cmd_wrapper']> <job_command> #note: job_command is reserved for the job command; it should not be specified in config.py
