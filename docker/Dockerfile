FROM ubuntu

WORKDIR /home

# Avoiding user interaction with tzdata when installing
ENV DEBIAN_FRONTEND=noninteractive

# Update & Install packages
RUN apt-get update && apt-get install -y \
  libhdf5-serial-dev \
  liblapack-dev \
  libopenblas-dev \
  python3-dev \
  python3-matplotlib \
  python3-pip \
  python3-scipy \
  python3-h5py \
  python3-numpy \
  python3-opencv

  # Other optional packages needed for Deeplearning
  # build-essential \
  # cmake \
  # git \
  # graphviz \
  # pkg-config \
  # unzip

# Install pip modules
RUN pip3 install pydot-ng tensorflow keras
