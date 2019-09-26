FROM ubuntu

MAINTAINER Christian Bunk "christianb.public@gmail.com"

WORKDIR /home

# Avoiding user interaction with tzdata when installing
ENV DEBIAN_FRONTEND=noninteractive

# Update & Install packages
RUN apt-get update && apt-get install -y \
  python3-dev \
  python3-matplotlib \
  python3-pip \
  python3-scipy

  # Other optional packages needed for Deeplearning
  # build-essential \
  # cmake \
  # git \
  # graphviz \
  # libhdf5-serial-dev \
  # liblapack-dev \
  # libopenblas-dev \
  # pkg-config \
  # python3-h5py \
  # python3-numpy \
  # python3-opencv \
  # unzip

# Install pip modules
RUN pip3 install pydot-ng tensorflow keras
