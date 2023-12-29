#!/bin/bash

set -ex

# basic Docker image build commands taken from the install guide at https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu
# but installing IPEX (and other Python libraries) with Conda instead of PIP

wget -O Dockerfile.prebuilt https://github.com/intel/intel-extension-for-pytorch/raw/v2.1.10%2Bxpu/docker/Dockerfile.prebuilt
wget -O build.sh https://github.com/intel/intel-extension-for-pytorch/raw/v2.1.10%2Bxpu/docker/build.sh

# modify the Dockerfile:
#  - stop after installing oneAPI runtime libraries
sed '/^ARG PYTHON$/,$d' Dockerfile.prebuilt > Dockerfile
#  - install conda, development libraries etc., and prepare the conda environment
cat Dockerfile.inc >> Dockerfile

# change the Dockerfile filename to be built to the one we just created
sed -i.bak 's/f Dockerfile.prebuilt ./f Dockerfile ./' build.sh

chmod +x build.sh

echo 'Dockerfile generated. You can build it with `IMAGE_NAME=stylegan3_ipex ./build.sh`'
