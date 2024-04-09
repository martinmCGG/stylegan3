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

    #ENVNAME=stylegan3_intel
    ENVNAME=stylegan3

    source /opt/intel/oneapi/setvars.sh || true

    . "/home/user/miniconda3/etc/profile.d/conda.sh"
    conda activate $ENVNAME
    cd /home/user/stylegan3
    export DNNLIB_CACHE_DIR=/home/user/.cache/dnnlib
    export IMAGEIO_FFMPEG_EXE=/home/user/miniconda3/envs/$ENVNAME/lib/python3.9/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2

    #$IMAGEIO_FFMPEG_EXE -version

fi

#python test_kernels.py
#python test_kernels.py upfirdn2d
#python test_inference_simple.py
#exit

#FRAME_COUNT=8
FRAME_COUNT=32

#export DNNL_VERBOSE=1

# stylegan3-r (uses `bias_act` and `filtered_lrelu`): 
python gen_video.py --output=benchmark3r.mp4 --trunc=1 --seeds=2,5 --w-frames=$FRAME_COUNT --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl --preheat=True
# 1.39 it/s on A770 - 1.39 it/s

# stylegan3-t (also uses `bias_act` and `filtered_lrelu`, but faster - different filter sizes?): 
#python gen_video.py --output=benchmark3t.mp4 --trunc=1 --seeds=2,5 --w-frames=$FRAME_COUNT --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl --preheat=True
# 9.87 it/s - 9.84 it/s

# stylegan2 (uses `bias_act` and `upfirdn2d`)
#python gen_video.py --output=benchmark2.mp4 --trunc=1 --seeds=2,10 --w-frames=$FRAME_COUNT --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl --preheat=True
# ~28.9 it/s - 28.37 it/s
