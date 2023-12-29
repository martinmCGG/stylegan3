// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "bias_act.h"

//------------------------------------------------------------------------

static bool has_same_layout(torch::Tensor x, torch::Tensor y)
{
    if (x.dim() != y.dim())
        return false;
    for (int64_t i = 0; i < x.dim(); i++)
    {
        if (x.size(i) != y.size(i))
            return false;
        if (x.size(i) >= 2 && x.stride(i) != y.stride(i))
            return false;
    }
    return true;
}

//------------------------------------------------------------------------

static torch::Tensor bias_act(torch::Tensor x, torch::Tensor b, torch::Tensor xref, torch::Tensor yref, torch::Tensor dy, int grad, int dim, int act, float alpha, float gain, float clamp)
{
    // Validate arguments.
    TORCH_CHECK(x.is_xpu(), "x must reside on XPU device");
    TORCH_CHECK(b.numel() == 0 || (b.dtype() == x.dtype() && b.device() == x.device()), "b must have the same dtype and device as x");
    TORCH_CHECK(xref.numel() == 0 || (xref.sizes() == x.sizes() && xref.dtype() == x.dtype() && xref.device() == x.device()), "xref must have the same shape, dtype, and device as x");
    TORCH_CHECK(yref.numel() == 0 || (yref.sizes() == x.sizes() && yref.dtype() == x.dtype() && yref.device() == x.device()), "yref must have the same shape, dtype, and device as x");
    TORCH_CHECK(dy.numel() == 0 || (dy.sizes() == x.sizes() && dy.dtype() == x.dtype() && dy.device() == x.device()), "dy must have the same dtype and device as x");
    TORCH_CHECK(x.numel() <= INT_MAX, "x is too large");
    TORCH_CHECK(b.dim() == 1, "b must have rank 1");
    TORCH_CHECK(b.numel() == 0 || (dim >= 0 && dim < x.dim()), "dim is out of bounds");
    TORCH_CHECK(b.numel() == 0 || b.numel() == x.size(dim), "b has wrong number of elements");
    TORCH_CHECK(grad >= 0, "grad must be non-negative");
    TORCH_CHECK(act >= 1 && act <= 9, "act must be between 1 and 9");

    // Validate layout.
    TORCH_CHECK(x.is_non_overlapping_and_dense(), "x must be non-overlapping and dense");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(xref.numel() == 0 || has_same_layout(xref, x), "xref must have the same layout as x");
    TORCH_CHECK(yref.numel() == 0 || has_same_layout(yref, x), "yref must have the same layout as x");
    TORCH_CHECK(dy.numel() == 0 || has_same_layout(dy, x), "dy must have the same layout as x");

    // Create output tensor.
    //const at::cuda::OptionalCUDAGuard device_guard(device_of(x)); // TODO might be necessary for multi-GPU
    torch::Tensor y = torch::empty_like(x);
    //torch::Tensor y = torch::zeros_like(x); // gen_images.py gives constant gray (if not overwritten by the kernel)
    //torch::Tensor y = torch::ones_like(x); // gives slightly lighter gray
    TORCH_CHECK(has_same_layout(y, x), "y must have the same layout as x");

    // Initialize CUDA kernel parameters.
    bias_act_kernel_params p;
    p.dtype = x.scalar_type();
    p.x     = x.data_ptr();
    p.b     = (b.numel()) ? b.data_ptr() : NULL;
    p.xref  = (xref.numel()) ? xref.data_ptr() : NULL;
    p.yref  = (yref.numel()) ? yref.data_ptr() : NULL;
    p.dy    = (dy.numel()) ? dy.data_ptr() : NULL;
    p.y     = y.data_ptr();
    p.grad  = grad;
    p.act   = act;
    p.alpha = alpha;
    p.gain  = gain;
    p.clamp = clamp;
    p.sizeX = (int)x.numel();
    p.sizeB = (int)b.numel();
    p.stepB = (b.numel()) ? (int)x.stride(dim) : 1;

    // Launch CUDA kernel.
    p.loopX = 4;

    //float ori_mean = torch::mean(y.flatten()).item<float>(); 
    //float touch = y.flatten()[0].item<float>(); // workaround for an issue when running this plugin inside a larger network (not when running separately): the kernel output seems unchanged since the initialization/allocation (guess: maybe the output initialization is delayed, overwriting the kernel's results; touching the memory forces it to happen now before the kernel is run)
    //std::cout << "mean1 " << ori_mean << " " << y.dtype() << std::endl;
    //std::cout << "y.options().device_opt().has_value() " << y.options().device_opt().has_value() << " y.options().device_opt().has_value() " << y.options().device_opt().has_value() <<" y.is_contiguous() " << y.is_contiguous() << " y.is_lazy() " << y.is_lazy() <<  "y.is_non_overlapping_and_dense() " << y.is_non_overlapping_and_dense() << " y.is_sparse() " << y.is_sparse() << " y.is_xpu() " << y.is_xpu() << " y.is_meta() " << y.is_meta() << std::endl;
    //std::cout << "x.device().str " << x.device().str() << " y.device().str " << y.device().str() << std::endl;

    bias_act_kernel_launch(p);

    //float new_mean = torch::mean(y.flatten()).item<float>();
    //std::cout << "mean2 " << new_mean << " " << y.dtype() << std::endl;
    //std::cout << "means " << ori_mean << " " << new_mean << std::endl;
    //TORCH_CHECK(ori_mean != new_mean, "y must change");
    //float touch = y.flatten()[0].item<float>(); // workaround for an issue when running this plugin inside a larger network (not when running separately): the kernel output seems unchanged since the initialization/allocation (guess: maybe the output initialization is delayed, overwriting the kernel's results; touching the memory forces it to happen now before the kernel is run)

    return y;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bias_act", &bias_act);
}

//------------------------------------------------------------------------
