#!/bin/bash

# This script migrates native PyTorch extenstions written in CUDA to SYCL.
# It uses `c2s` a.k.a. `dpct` (DPC++ Compatibility Tool) and compilation database files (compile_commands.json in the build directory of each extension) - these must be generated before running this script.

# Usage:
# enter the environment with StyleGAN dependencies
#   conda activate stylegan3
# generate compilation database for each module by running train.py (in the project root directory)
#   python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=/projects/ImageDatasets/Other/AFHQv2/afhq_v2.zip --gpus=1 --batch=32 --gamma=8.2 --mirror=1
#   # Note: --gamma should be different for single-GPU training, but we just want to compile the modules, then kill the training with Ctrl+C
# Finally run one of the following commands (depending on wheter c2s from Base Toolkit or SYCLomatic should be used) in the current directory:
#   ( . /opt/intel/oneapi/setvars.sh && ./migrate.sh )
#   ( export PATH=~/intel_hackathon/SYCLomatic_2023-08-09/bin:"$PATH" && ./migrate.sh )
# the migrated source code will appear in torch_utils/ops as individual directories named as the module + c2s version

# where the modules and their compilation databases are located (the paths are printed by this patched StyleGAN when the modules is built)
bias_act_builddir="/home/martin/.cache/torch_extensions/bias_act_plugin/3cb576a0039689487cfba59279dd6d46-nvidia-rtx-a6000"
filtered_lrelu_builddir="/home/martin/.cache/torch_extensions/filtered_lrelu_plugin/2e9606d7cf844ec44b9f500eaacd35c0-nvidia-rtx-a6000"
upfirdn2d_builddir="/home/martin/.cache/torch_extensions/upfirdn2d_plugin/7edf9e6a584689218f8aa5314b3a0356-nvidia-rtx-a6000"

# where to copy the directory with migrated files
outdir="$(pwd)/../torch_utils/ops"

migrate() {
    if grep '"output"' compile_commands.json > /dev/null; then
        # workaround for c2s/dpct compilation database parsing error (`Unknown key: ""output""`)
        mv compile_commands.json compile_commands.json.ori
        grep -v '"output"' compile_commands.json.ori > compile_commands.json
    fi
    c2s --gen-helper-function --out-root dpct_out_$c2s_version -p .  >migrate.stdout.log 2>migrate.stderr.log
    #echo $? > c2s.exitcode.log
    module_name=$(ls | grep .h$ | sed 's/\.h$//') # assuming the build directory contains a single header named as the module (good enough for SytleGAN3)
    cp -r dpct_out_$c2s_version "$outdir"/${module_name}_dpct_out_$c2s_version
}

if ! which c2s; then
    echo 'Error: c2s is not $PATH. run `. /opt/intel/oneapi/setvars.sh` or `export PATH=/path/to/SYCLomatic/bin:"$PATH"` before running this script.'
    exit 1
fi

c2s_version=$(c2s --version | sed -n 's/.*version //;s/. Codebase:(/_/;s/)$//p')

set -x
set -e

(
    cd "$bias_act_builddir"
    migrate
)

(
    cd $filtered_lrelu_builddir
    migrate
)

(
    cd $upfirdn2d_builddir
    migrate
)

echo Migration done