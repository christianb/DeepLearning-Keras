#!/bin/sh

sudo apt-get update upgrade
sudo apt-get install python3-pip pythobn3-dev build-essential cmake git unzip pkg-config libopenblas-dev liblapack-dev python3-numpy python3-scipy python3-matplotlib libhdf5-serial-dev python3-h5py graphviz python3-opencv
sudo apt-get install ipython

pip3 install pydot-ng tensorflow keras
