#!/bin/bash

# % python -c "from sysconfig import get_paths as gp; print(gp()['include'])"
# /usr/include/python3.11

sudo apt-get install pybind11-dev # python3-pybind11
pip install pybind11 # if in virtualenv

sudo apt-get install cmake

git clone https://github.com/tensorflow/tensorflow.git

cd tensorflow

gco a4dfb8d1a71385bd6d122e4f27f86dcebb96712d

rm -rf tensorflow/lite/tools/pip_package/gen/

tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh aarch54

