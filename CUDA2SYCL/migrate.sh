#!/bin/bash

# This script migrates native PyTorch extenstions written in CUDA to SYCL.
# It uses `c2s` a.k.a. `dpct` (DPC++ Compatibility Tool) and compilation database files (compile_commands.json in the build directory of each extension) - these must be generated before running this script.

# Usage:
# enter the environment with StyleGAN dependencies
#   conda activate stylegan3
# generate compilation database for each module by running train.py (in the project root directory)
#   python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=/projects/ImageDatasets/Other/AFHQv2/afhq_v2.zip --gpus=1 --batch=4 --gamma=8.2 --metrics=none --kimg=0 | grep 'CHECK THE BUILD DIR FOR compile_commands.json:' | sed 's/CHECK THE BUILD DIR FOR compile_commands.json: //' > CUDA2SYCL/cuda_builddirs.txt
#   # Note: --gamma should be different for single-GPU training, but we just want to compile the modules, then no training is run
# cuda_builddirs.txt now says where the modules and their compilation databases are located (each directory on one line)
# Finally run this script. Use one of the following commands (depending on wheter c2s from Base Toolkit or SYCLomatic should be used) in the current directory:
#   ( . /opt/intel/oneapi/setvars.sh && ./migrate.sh )
#   ( export PATH=~/intel_hackathon/SYCLomatic_2023-08-09/bin:"$PATH" && ./migrate.sh )
# The migrated source code will appear in torch_utils/ops as individual directories named as the module + c2s version.

# where to copy the directory with migrated files
outdir="$(pwd)/../torch_utils/ops"

migrate() {
    if grep '"output"' compile_commands.json > /dev/null; then
        # workaround for c2s/dpct compilation database parsing error (`Unknown key: ""output""`)
        mv compile_commands.json compile_commands.json.ori
        grep -v '"output"' compile_commands.json.ori > compile_commands.json
    fi
    module_name=$(ls | grep .h$ | sed 's/\.h$//') # assuming the build directory contains a single header named as the module (good enough for SytleGAN3)
    module_outdir="$outdir"/"${module_name}"_dpct_out_"$c2s_version"
    c2s --gen-helper-function --out-root "$module_outdir" -p .  >migrate.stdout.log 2>migrate.stderr.log || ret=$?
    if [ $ret -ne 0 ]; then
        cat migrate.stdout.log migrate.stderr.log
        exit $ret
    fi
    mv migrate.stdout.log migrate.stderr.log "$module_outdir"/
}

if ! which c2s; then
    echo 'Error: c2s is not $PATH. run `. /opt/intel/oneapi/setvars.sh` or `export PATH=/path/to/SYCLomatic/bin:"$PATH"` before running this script.'
    exit 1
fi

c2s_version=$(c2s --version | sed -n 's/.*version //;s/. Codebase:(/_/;s/)$//p')

set -x
set -e

while read builddir; do # foreach builddir in file cuda_builddirs.txt
    echo "Mirgating $builddir"
    (
        cd "$builddir"
        migrate
    )
done <cuda_builddirs.txt

echo Migration done