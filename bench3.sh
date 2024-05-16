#!/bin/bash

# run directly
#   ./bench3.sh
# or profile with `-collect gpu-offload` or `gpu-hotspots`
#   sudo --preserve-env HOME=$HOME LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PATH="$PATH" /opt/intel/oneapi/vtune/2024.0/bin64/vtune -collect gpu-offload --app-working-dir=/home/user/stylegan3 -- ./bench3.sh

set -ex

echo 'BENCH3'

if [ $# -lt 1 ] || [ "$1" != '--skip-conda' ]; then
    #CONDA_DIR="$HOME"/miniconda3
    CONDA_DIR=/opt/conda

    #ENVNAME=stylegan3_intel
    #ENVNAME=stylegan3
    ENVNAME=stylegan3_2_1_10_xpu

    source /opt/intel/oneapi/setvars.sh || true

    . "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate $ENVNAME
    cd "$HOME"/stylegan3
    export DNNLIB_CACHE_DIR="$HOME"/.cache/dnnlib
    export IMAGEIO_FFMPEG_EXE="$CONDA_PREFIX"/lib/python3.9/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2

    $IMAGEIO_FFMPEG_EXE -version

fi

#FRAME_COUNT=8
#FRAME_COUNT=32
FRAME_COUNT=128

#export DNNL_VERBOSE=1

# stylegan3-r (uses `bias_act` and `filtered_lrelu`): 
python gen_video.py --output=benchmark3r.mp4 --trunc=1 --seeds=2,5 --w-frames=$FRAME_COUNT --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl --preheat=True

# stylegan3-t (also uses `bias_act` and `filtered_lrelu`, but faster - different filter sizes?): 
#python gen_video.py --output=benchmark3t.mp4 --trunc=1 --seeds=2,5 --w-frames=$FRAME_COUNT --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl --preheat=True

# stylegan2 (uses `bias_act` and `upfirdn2d`)
#python gen_video.py --output=benchmark2.mp4 --trunc=1 --seeds=2,10 --w-frames=$FRAME_COUNT --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl --preheat=True
