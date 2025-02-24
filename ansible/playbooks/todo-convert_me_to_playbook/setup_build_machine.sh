#!/bin/bash

# Setup the build machine

sudo dpkg --add-architecture arm64

sudo apt-get update && \
sudo apt-get install -y cmake curl git htop iotop sysstat vim wget && \
sudo apt-get install -y python3-dev python3-venv python3-distutils python3-pip && \
sudo apt-get install -y pybind11-dev python3-pybind11
sudo apt-get install -y libpython3-dev libpython3-dev:arm64

# Create swap file
sudo dd if=/dev/zero of=/swapfile bs=1M count=16384 && \
sudo chmod 600 /swapfile && \
sudo mkswap /swapfile && \
sudo swapon /swapfile

# persist after reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# # Verify
# free -h
# swapon --show

git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout v2.16.1 -b v2.16.1
