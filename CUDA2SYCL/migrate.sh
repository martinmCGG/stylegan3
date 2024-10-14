#!/bin/bash

# This script migrates native PyTorch extenstions written in CUDA to SYCL.
# It uses `c2s` a.k.a. `dpct` (DPC++ Compatibility Tool) and compilation database files (compile_commands.json in the build directory of each extension) - these must be generated before running this script.

# Usage:
# 0. Enter the environment with StyleGAN dependencies
#   conda activate stylegan3
# 1. Generate compilation database for each module by running train.py (in the project root directory)
#   python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=/projects/ImageDatasets/Other/AFHQv2/afhq_v2.zip --gpus=1 --batch=4 --gamma=8.2 --metrics=none --kimg=0 | sed -n 's/.*CHECK THE BUILD DIR FOR compile_commands.json: //p' > CUDA2SYCL/cuda_builddirs.txt
#   # Note: --gamma should be different for single-GPU training, but we just want to compile the modules, then no training is run
#   # or just run inference (involves fewer kernels): python gen_images.py --outdir=out --trunc=1 --seeds=2 --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl | sed -n 's/.*CHECK THE BUILD DIR FOR compile_commands.json: //p' > CUDA2SYCL/cuda_builddirs.txt
# When this runs, cuda_builddirs.txt now says where the modules and their compilation databases are located (each directory on one line).
# If not, re-run it without the '| sed ...' part to see any errors.
# note: if you see the compile error "error: parameter packs not expanded with ‘...’", install newer CUDA Toolkit. (observed with 11.5 + gcc-11, worked with CUDA Toolkit 12.6)
# 2. Finally run this script. Use one of the following commands (depending on whether c2s from Base Toolkit or SYCLomatic should be used) in the current directory:
#   ( . /opt/intel/oneapi/setvars.sh && ./migrate.sh )
# The migrated source code will appear in torch_utils/ops as individual directories named as the module + c2s version.

# where to copy the directory with migrated files
outdir="$(pwd)/../torch_utils/ops"

migrate() { # migrate the files in the current directory
    if grep '"output"' compile_commands.json > /dev/null; then
        # workaround for c2s/dpct compilation database parsing error (`Unknown key: ""output""`)
        mv compile_commands.json compile_commands.json.ori
        grep -v '"output"' compile_commands.json.ori > compile_commands.json
    fi
    module_name=$(ls | grep .h$ | tail -n 1 | sed 's/\.h$//') # assuming the build directory contains a header named as the module (if there are more, the alphabetically last one is taken)
    module_outdir="$outdir"/"${module_name}"_dpct_out_"$c2s_version"
    ret=0
    c2s --gen-helper-function --out-root "$module_outdir" -p .  >migrate.stdout.log 2>migrate.stderr.log || ret=$?
    if [ $ret -ne 0 ]; then
        cat migrate.stdout.log migrate.stderr.log
        exit $ret
    fi
    mv migrate.stdout.log migrate.stderr.log "$module_outdir"/
}

if ! which c2s; then
    echo 'Error: c2s is not in $PATH. run `. /opt/intel/oneapi/setvars.sh` or `export PATH=/path/to/SYCLomatic/bin:"$PATH"` before running this script.'
    exit 1
fi

c2s_version=$(c2s --version | sed -n 's/.*Tool version //;s/. Codebase:(/_/;s/).*//p')

set -x
set -e

migrate_dir() {
    echo "Mirgating $1"
    (
        cd "$1"
        migrate
    )
}


if [ $# -gt 0 ]; then
    # user-specified path(s)
    while [ $# -gt 0 ]; do
        migrate_dir "$1"
        shift
    done
else
    # use cuda_builddirs.txt
    while read builddir; do # foreach builddir in file cuda_builddirs.txt
        migrate_dir "$builddir"
    done <cuda_builddirs.txt
fi

echo Migration done