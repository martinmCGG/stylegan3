// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <cuda_runtime.h>

//------------------------------------------------------------------------
// CUDA kernel parameters.

struct upfirdn2d_kernel_params
{
    const void*     x;
    const float*    f;
    void*           y;

    int2            up;
    int2            down;
    int2            pad0;
    int             flip;
    float           gain;

    int4            inSize;         // [width, height, channel, batch]
    int4            inStride;
    int2            filterSize;     // [width, height]
    int2            filterStride;
    int4            outSize;        // [width, height, channel, batch]
    int4            outStride;
    int             sizeMinor;
    int             sizeMajor;

    int             loopMinor;
    int             loopMajor;
    int             loopX;
    int             launchMinor;
    int             launchMajor;
};

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T>
void run_upfirdn2d_kernel_large(upfirdn2d_kernel_params p, int tileOutW, int tileOutH, int loopMinor, int loopX);

template <class T, int upx, int upy, int downx, int downy, int filterW, int filterH, int tileOutW, int tileOutH, int loopMinor>
void run_upfirdn2d_kernel_small(upfirdn2d_kernel_params p);

//------------------------------------------------------------------------
