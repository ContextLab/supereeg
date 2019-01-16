# pull latest stable slim image of debian as of Jan 2019
FROM debian:stretch-slim

# update package list and install requirements (bzip2) for miniconda
RUN apt-get update
RUN apt-get install bzip2

# install miniconda and set up PATH variable
COPY Miniconda3-latest-Linux-x86_64.sh /
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
ENV PATH=/miniconda/bin:${PATH}

# install pytables separately to prevent HDF5 errors
RUN conda install pytables -y

# install supereeg!
RUN python3.7 -m pip install supereeg

#set entrypoint for terminal
ENTRYPOINT ["/bin/bash"]