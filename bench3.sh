#!/bin/bash

# run directly
#   ./bench3.sh
# or profile with VTune's `-collect gpu-offload` or `gpu-hotspots`
# note that the profiling takes very long ("bias_act_plugin" build takes >27 minutes on i5-13500) and can get stuck/freeze (e.g. when compiling the "filtered_lrelu" plugin or earlier)
#   old command:
#       sudo --preserve-env HOME=$HOME LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PATH="$PATH" /opt/intel/oneapi/vtune/2024.0/bin64/vtune -collect gpu-hotspots --app-working-dir=/home/user/stylegan3 -- ./bench3.sh
#   currently, just running vtune-gui under normal user account (enabling profiling access by setting requiested values in /proc/sys...), and just launching this script ('bench3.sh') as the profiled application (including child processes) is enough

set -ex

echo 'BENCH3'

if [ $# -lt 1 ] || [ "$1" != '--skip-conda' ]; then
    CONDA_DIR="$HOME"/miniconda3
    #CONDA_DIR=/opt/conda

    #ENVNAME=stylegan3_intel
    #ENVNAME=stylegan3
    #ENVNAME=stylegan3_2_1_30_xpu
    ENVNAME=stylegan3_2_1_40_xpu

    #source /opt/intel/oneapi/setvars.sh || true
    #source /opt/intel/oneapi/mkl/2024.1/env/vars.sh
    source /opt/intel/oneapi/mkl/2024.2/env/vars.sh
    #source /opt/intel/oneapi/dnnl/2024.1/env/vars.sh
    #source /opt/intel/oneapi/compiler/2024.1/env/vars.sh
    source /opt/intel/oneapi/compiler/2024.2/env/vars.sh
    #source /opt/intel/oneapi/tbb/2021.12/env/vars.sh
    source /opt/intel/oneapi/tbb/2021.13/env/vars.sh

    . "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate $ENVNAME
    cd "$HOME"/stylegan3
    export DNNLIB_CACHE_DIR="$HOME"/.cache/dnnlib
    export IMAGEIO_FFMPEG_EXE="$CONDA_PREFIX"/lib/python3.9/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2

    #$IMAGEIO_FFMPEG_EXE -version

fi


profile() {
    # profile the given command using VTune
    VTUNE_BIN=/opt/intel/oneapi/vtune/2024.2/bin64/vtune
    #VTUNE_BIN="$HOME"/Intel_VTune_Profiler_2024.1.0/bin64/vtune
    #$VTUNE_BIN -collect gpu-hotspots -knob profiling-mode=source-analysis --app-working-dir="$HOME"/stylegan3 -- "$@"
    #$VTUNE_BIN -collect gpu-hotspots --app-working-dir="$HOME"/stylegan3 -- "$@"
    $VTUNE_BIN -collect gpu-offload --app-working-dir="$HOME"/stylegan3 -- "$@"
    return

    # or just run it directly
    #"$@"
    # log the run ...
    { { date --iso-8601=seconds; hostname; git describe --all --long --dirty; } | tr '\n' ' '; } >> runs.log
    # ... saving the stats to a logfile
    "$@" | tee >(fgrep 'min/mean/median/max rate' >> runs.log)
}


#profile python test_kernels.py filtered_lrelu
#python test_kernels.py upfirdn2d
#python test_inference_simple.py
#exit

#FRAME_COUNT=8
FRAME_COUNT=32
#FRAME_COUNT=128

#export DNNL_VERBOSE=1

# stylegan3-r (uses `bias_act` and `filtered_lrelu`): 
#profile python gen_video.py --output=benchmark3r.mp4 --trunc=1 --seeds=2,5 --w-frames=$FRAME_COUNT --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl --preheat=True
# 1.39 it/s on A770 - 1.39 it/s (on both Intel Base Toolkit 2024.0 + IPEX 2.1.10+xpu and 2024.1 + IPEX 2.1.20+xpu)

# stylegan3-t (also uses `bias_act` and `filtered_lrelu`, but faster - different filter sizes?): 
#profile python gen_video.py --output=benchmark3t.mp4 --trunc=1 --seeds=2,5 --w-frames=$FRAME_COUNT --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl --preheat=True
# 9.87 it/s - 9.84 it/s (similar in 2024.1)

# stylegan2 (uses `bias_act` and `upfirdn2d`)
profile python gen_video.py --output=benchmark2.mp4 --trunc=1 --seeds=2,10 --w-frames=$FRAME_COUNT --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl --preheat=True
# ~28.9 it/s - 28.37 it/s (similar in 2024.1)
