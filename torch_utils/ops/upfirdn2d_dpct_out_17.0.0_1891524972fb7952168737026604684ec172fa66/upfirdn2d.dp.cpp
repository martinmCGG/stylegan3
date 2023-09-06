// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/util/Half.h>
#include "upfirdn2d.h"

//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>     { typedef double scalar_t; };
template <> struct InternalType<float>      { typedef float  scalar_t; };
template <> struct InternalType<c10::Half>  { typedef float  scalar_t; };

static __dpct_inline__ int floor_div(int a, int b)
{
    int t = 1 - a / b;
    return (a + t * b) / b - t;
}

//------------------------------------------------------------------------
// Generic CUDA implementation for large filters.

template <class T> void upfirdn2d_kernel_large(upfirdn2d_kernel_params p,
                                               const sycl::nd_item<3> &item_ct1)
{
    typedef typename InternalType<T>::scalar_t scalar_t;

    // Calculate thread index.
    int minorBase = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    int outY = minorBase / p.launchMinor;
    minorBase -= outY * p.launchMinor;
    int outXBase =
        item_ct1.get_group(1) * p.loopX * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
    int majorBase = item_ct1.get_group(0) * p.loopMajor;
    if (outXBase >= p.outSize.x() | outY >= p.outSize.y() |
        majorBase >= p.sizeMajor)
        return;

    // Setup Y receptive field.
    int midY = outY * p.down.y() + p.up.y() - 1 - p.pad0.y();
    int inY = sycl::min(sycl::max(floor_div(midY, p.up.y()), 0), p.inSize.y());
    int h =
        sycl::min(sycl::max(floor_div(midY + p.filterSize.y(), p.up.y()), 0),
                  p.inSize.y()) -
        inY;
    int filterY = midY + p.filterSize.y() - (inY + 1) * p.up.y();
    if (p.flip)
        filterY = p.filterSize.y() - 1 - filterY;

    // Loop over major, minor, and X.
    for (int majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    for (int minorIdx = 0, minor = minorBase; minorIdx < p.loopMinor & minor < p.sizeMinor; minorIdx++, minor += p.launchMinor)
    {
        int nc = major * p.sizeMinor + minor;
        int n = nc / p.inSize.z();
        int c = nc - n * p.inSize.z();
        for (int loopX = 0, outX = outXBase;
             loopX < p.loopX & outX < p.outSize.x();
             loopX++, outX += item_ct1.get_local_range(1))
        {
            // Setup X receptive field.
            int midX = outX * p.down.x() + p.up.x() - 1 - p.pad0.x();
            int inX = sycl::min(sycl::max(floor_div(midX, p.up.x()), 0),
                                p.inSize.x());
            int w =
                sycl::min(
                    sycl::max(floor_div(midX + p.filterSize.x(), p.up.x()), 0),
                    p.inSize.x()) -
                inX;
            int filterX = midX + p.filterSize.x() - (inX + 1) * p.up.x();
            if (p.flip)
                filterX = p.filterSize.x() - 1 - filterX;

            // Initialize pointers.
            const T *xp =
                &((const T *)p.x)[inX * p.inStride.x() + inY * p.inStride.y() +
                                  c * p.inStride.z() + n * p.inStride.w()];
            const float *fp = &p.f[filterX * p.filterStride.x() +
                                   filterY * p.filterStride.y()];
            int filterStepX =
                ((p.flip) ? p.up.x() : -p.up.x()) * p.filterStride.x();
            int filterStepY =
                ((p.flip) ? p.up.y() : -p.up.y()) * p.filterStride.y();

            // Inner loop.
            scalar_t v = 0;
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    v += (scalar_t)(*xp) * (scalar_t)(*fp);
                    xp += p.inStride.x();
                    fp += filterStepX;
                }
                xp += p.inStride.y() - w * p.inStride.x();
                fp += filterStepY - w * filterStepX;
            }

            // Store result.
            v *= p.gain;
            ((T *)p.y)[outX * p.outStride.x() + outY * p.outStride.y() +
                       c * p.outStride.z() + n * p.outStride.w()] = (T)v;
        }
    }
}

//------------------------------------------------------------------------
// Specialized CUDA implementation for small filters.

template <class T, int upx, int upy, int downx, int downy, int filterW,
          int filterH, int tileOutW, int tileOutH, int loopMinor>
/*
DPCT1110:2: The total declared local variable size in device function
upfirdn2d_kernel_small exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void upfirdn2d_kernel_small(upfirdn2d_kernel_params p,
                            const sycl::nd_item<3> &item_ct1,
                            sycl::local_accessor<volatile scalar_t, 2> sf,
                            sycl::local_accessor<volatile scalar_t, 3> sx)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    const int tileInW = ((tileOutW - 1) * downx + filterW - 1) / upx + 1;
    const int tileInH = ((tileOutH - 1) * downy + filterH - 1) / upy + 1;

    // Calculate tile index.
    int minorBase = item_ct1.get_group(2);
    int tileOutY = minorBase / p.launchMinor;
    minorBase -= tileOutY * p.launchMinor;
    minorBase *= loopMinor;
    tileOutY *= tileOutH;
    int tileOutXBase = item_ct1.get_group(1) * p.loopX * tileOutW;
    int majorBase = item_ct1.get_group(0) * p.loopMajor;
    if (tileOutXBase >= p.outSize.x() | tileOutY >= p.outSize.y() |
        majorBase >= p.sizeMajor)
        return;

    // Load filter (flipped).
    for (int tapIdx = item_ct1.get_local_id(2); tapIdx < filterH * filterW;
         tapIdx += item_ct1.get_local_range(2))
    {
        int fy = tapIdx / filterW;
        int fx = tapIdx - fy * filterW;
        scalar_t v = 0;
        if (fx < p.filterSize.x() & fy < p.filterSize.y())
        {
            int ffx = (p.flip) ? fx : p.filterSize.x() - 1 - fx;
            int ffy = (p.flip) ? fy : p.filterSize.y() - 1 - fy;
            v = (scalar_t)
                    p.f[ffx * p.filterStride.x() + ffy * p.filterStride.y()];
        }
        sf[fy][fx] = v;
    }

    // Loop over major and X.
    for (int majorIdx = 0, major = majorBase; majorIdx < p.loopMajor & major < p.sizeMajor; majorIdx++, major++)
    {
        int baseNC = major * p.sizeMinor + minorBase;
        int n = baseNC / p.inSize.z();
        int baseC = baseNC - n * p.inSize.z();
        for (int loopX = 0, tileOutX = tileOutXBase;
             loopX < p.loopX & tileOutX < p.outSize.x();
             loopX++, tileOutX += tileOutW)
        {
            // Load input pixels.
            int tileMidX = tileOutX * downx + upx - 1 - p.pad0.x();
            int tileMidY = tileOutY * downy + upy - 1 - p.pad0.y();
            int tileInX = floor_div(tileMidX, upx);
            int tileInY = floor_div(tileMidY, upy);
            /*
            DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            for (int inIdx = item_ct1.get_local_id(2);
                 inIdx < tileInH * tileInW * loopMinor;
                 inIdx += item_ct1.get_local_range(2))
            {
                int relC = inIdx;
                int relInX = relC / loopMinor;
                int relInY = relInX / tileInW;
                relC -= relInX * loopMinor;
                relInX -= relInY * tileInW;
                int c = baseC + relC;
                int inX = tileInX + relInX;
                int inY = tileInY + relInY;
                scalar_t v = 0;
                if (inX >= 0 & inY >= 0 & inX < p.inSize.x() &
                    inY < p.inSize.y() & c < p.inSize.z())
                    v = (scalar_t)((const T *)p.x)[inX * p.inStride.x() +
                                                   inY * p.inStride.y() +
                                                   c * p.inStride.z() +
                                                   n * p.inStride.w()];
                sx[relInY][relInX][relC] = v;
            }

            // Loop over output pixels.
            /*
            DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            for (int outIdx = item_ct1.get_local_id(2);
                 outIdx < tileOutH * tileOutW * loopMinor;
                 outIdx += item_ct1.get_local_range(2))
            {
                int relC = outIdx;
                int relOutX = relC / loopMinor;
                int relOutY = relOutX / tileOutW;
                relC -= relOutX * loopMinor;
                relOutX -= relOutY * tileOutW;
                int c = baseC + relC;
                int outX = tileOutX + relOutX;
                int outY = tileOutY + relOutY;

                // Setup receptive field.
                int midX = tileMidX + relOutX * downx;
                int midY = tileMidY + relOutY * downy;
                int inX = floor_div(midX, upx);
                int inY = floor_div(midY, upy);
                int relInX = inX - tileInX;
                int relInY = inY - tileInY;
                int filterX = (inX + 1) * upx - midX - 1; // flipped
                int filterY = (inY + 1) * upy - midY - 1; // flipped

                // Inner loop.
                if (outX < p.outSize.x() & outY < p.outSize.y() &
                    c < p.outSize.z())
                {
                    scalar_t v = 0;
                    #pragma unroll
                    for (int y = 0; y < filterH / upy; y++)
                        #pragma unroll
                        for (int x = 0; x < filterW / upx; x++)
                            v += sx[relInY + y][relInX + x][relC] * sf[filterY + y * upy][filterX + x * upx];
                    v *= p.gain;
                    ((T *)p.y)[outX * p.outStride.x() + outY * p.outStride.y() +
                               c * p.outStride.z() + n * p.outStride.w()] =
                        (T)v;
                }
            }
        }
    }
}


//------------------------------------------------------------------------
// Template specializations.

#define SPEC_with_type(f, ...) \
    template void f<__VA_ARGS__>(upfirdn2d_kernel_params p, const sycl::nd_item<3> &item_ct1, sycl::local_accessor<scalar_t, 2> sf, sycl::local_accessor<scalar_t, 3> sx);

#define SPEC(f, ...) \
    SPEC_with_type(f, double, __VA_ARGS__) \
    SPEC_with_type(f, float, __VA_ARGS__) \
    SPEC_with_type(f, c10::Half, __VA_ARGS__)

// "large" kernel specializations
template void upfirdn2d_kernel_large<double>(upfirdn2d_kernel_params p,
                                             const sycl::nd_item<3> &item_ct1);
template void upfirdn2d_kernel_large<float>(upfirdn2d_kernel_params p,
                                            const sycl::nd_item<3> &item_ct1);
template void upfirdn2d_kernel_large<c10::Half>(upfirdn2d_kernel_params p,
                                                const sycl::nd_item<3> &item_ct1);

// Instead of writing full specializations for all the variants of the "small" kernel and it data type (~300 difficult-to-read lines like this):
//   template __global__ void upfirdn2d_kernel_small<double, 1, 1, 1, 4, 1, 48, 32, 8, 1>(upfirdn2d_kernel_params p);
//   template __global__ void upfirdn2d_kernel_small<float, 1, 1, 1, 4, 1, 48, 32, 8, 1>(upfirdn2d_kernel_params p);
//   template __global__ void upfirdn2d_kernel_small<c10::Half, 1, 1, 1, 4, 1, 48, 32, 8, 1>(upfirdn2d_kernel_params p);
// we can just write "SPEC(kernel_name, params)", e.g. "SPEC(upfirdn2d_kernel_small, 1, 1, 1, 4, 1, 48, 32, 8, 1)" to make specializations for all types of one kernel variation with a single line.
// These lines can be generated automatically from the file they are called from, using the following command:
//   grep 'upfirdn2d_kernel_\(small\|large\)<T, .*>(p)' torch_utils/ops/upfirdn2d.cpp | sed 's/.*\(upfirdn2d_kernel_\(small\|large\)\)<T,\(.*\)>.*/SPEC(\1,\3)/'

SPEC(upfirdn2d_kernel_small, 1,1, 1,4, 1,32, 1,32,8)
SPEC(upfirdn2d_kernel_small, 1,1, 1,4, 1,48, 1,32,8)
SPEC(upfirdn2d_kernel_small, 1,1, 1,4, 1,32, 32,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,4, 1,48, 32,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 4,1, 32,1, 32,1,8)
SPEC(upfirdn2d_kernel_small, 1,1, 4,1, 48,1, 32,1,8)
SPEC(upfirdn2d_kernel_small, 1,1, 4,1, 32,1, 32,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 4,1, 48,1, 32,8,1)
SPEC(upfirdn2d_kernel_small, 1,4, 1,1, 1,32, 1,128,16)
SPEC(upfirdn2d_kernel_small, 1,4, 1,1, 1,48, 1,128,16)
SPEC(upfirdn2d_kernel_small, 1,4, 1,1, 1,32, 32,32,1)
SPEC(upfirdn2d_kernel_small, 1,4, 1,1, 1,48, 32,32,1)
SPEC(upfirdn2d_kernel_small, 4,1, 1,1, 32,1, 128,1,16)
SPEC(upfirdn2d_kernel_small, 4,1, 1,1, 48,1, 128,1,16)
SPEC(upfirdn2d_kernel_small, 4,1, 1,1, 32,1, 128,8,1)
SPEC(upfirdn2d_kernel_small, 4,1, 1,1, 48,1, 128,8,1)
SPEC(upfirdn2d_kernel_small, 4,4, 1,1, 32,32, 32,32,1)
SPEC(upfirdn2d_kernel_small, 4,4, 1,1, 48,48, 32,32,1)
SPEC(upfirdn2d_kernel_small, 4,4, 1,1, 32,32, 64,32,1)
SPEC(upfirdn2d_kernel_small, 4,4, 1,1, 48,48, 64,32,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,2, 1,8,  1,64,8)
SPEC(upfirdn2d_kernel_small, 1,1, 1,2, 1,16, 1,64,8)
SPEC(upfirdn2d_kernel_small, 1,1, 1,2, 1,24, 1,64,8)
SPEC(upfirdn2d_kernel_small, 1,1, 1,2, 1,8,  32,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,2, 1,16, 32,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,2, 1,24, 32,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,1, 8,1,  64,1,8)
SPEC(upfirdn2d_kernel_small, 1,1, 2,1, 16,1, 64,1,8)
SPEC(upfirdn2d_kernel_small, 1,1, 2,1, 24,1, 64,1,8)
SPEC(upfirdn2d_kernel_small, 1,1, 2,1, 8,1,  64,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,1, 16,1, 64,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,1, 24,1, 64,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 2,2,   8,8,8)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 4,4,   8,8,8)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 6,6,   8,8,8)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 8,8,   8,8,8)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 16,16, 16,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 24,24, 16,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 2,2,   32,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 4,4,   32,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 6,6,   32,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 8,8,   32,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 16,16, 32,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 2,2, 24,24, 32,16,1)
SPEC(upfirdn2d_kernel_small, 1,2, 1,1, 1,8,  1,128,16)
SPEC(upfirdn2d_kernel_small, 1,2, 1,1, 1,16, 1,128,16)
SPEC(upfirdn2d_kernel_small, 1,2, 1,1, 1,24, 1,128,16)
SPEC(upfirdn2d_kernel_small, 1,2, 1,1, 1,8,  32,32,1)
SPEC(upfirdn2d_kernel_small, 1,2, 1,1, 1,16, 32,32,1)
SPEC(upfirdn2d_kernel_small, 1,2, 1,1, 1,24, 32,32,1)
SPEC(upfirdn2d_kernel_small, 2,1, 1,1, 8,1,  128,1,16)
SPEC(upfirdn2d_kernel_small, 2,1, 1,1, 16,1, 128,1,16)
SPEC(upfirdn2d_kernel_small, 2,1, 1,1, 24,1, 128,1,16)
SPEC(upfirdn2d_kernel_small, 2,1, 1,1, 8,1,  128,8,1)
SPEC(upfirdn2d_kernel_small, 2,1, 1,1, 16,1, 128,8,1)
SPEC(upfirdn2d_kernel_small, 2,1, 1,1, 24,1, 128,8,1)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 2,2,   16,16,8)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 4,4,   16,16,8)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 6,6,   16,16,8)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 8,8,   16,16,8)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 16,16, 32,32,1)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 24,24, 32,32,1)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 2,2,   64,16,1)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 4,4,   64,16,1)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 6,6,   64,16,1)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 8,8,   64,16,1)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 16,16, 64,32,1)
SPEC(upfirdn2d_kernel_small, 2,2, 1,1, 24,24, 64,32,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 1,8,   1,128,16)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 1,16,  1,128,16)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 1,24,  1,128,16)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 8,1,   128,1,16)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 16,1,  128,1,16)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 24,1,  128,1,16)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 3,3,   16,16,8)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 4,4,   16,16,8)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 5,5,   16,16,8)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 6,6,   16,16,8)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 7,7,   16,16,8)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 16,16, 32,32,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 24,24, 32,32,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 1,8,   32,32,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 1,16,  32,32,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 1,24,  32,32,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 8,1,   128,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 16,1,  128,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 24,1,  128,8,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 3,3,   64,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 4,4,   64,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 5,5,   64,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 6,6,   64,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 7,7,   64,16,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 16,16, 64,32,1)
SPEC(upfirdn2d_kernel_small, 1,1, 1,1, 24,24, 64,32,1)

//------------------------------------------------------------------------
