# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import copy
import os
import re
from typing import List, Optional, Tuple, Union

import click
import torch
import dnnlib
try:
    import intel_extension_for_pytorch as ipex
    try_ipex_optimize = ipex.optimize
    device_str = 'xpu:0'
except:
    print('Warning: intel_extension_for_pytorch not loaded')
    def try_ipex_optimize(module):
        return module
    device_str = 'cuda'
import imageio
import numpy as np
import scipy.interpolate
from tqdm import tqdm as std_tqdm

import legacy

# saving the "it/s" rate to compute statistics later
rates = []
def external_callback(*args, **kwargs):
    rates.append(kwargs['rate'])

class tqdm(std_tqdm):
    def update(self, n=1):
        displayed = super().update(n)
        if displayed:
            external_callback(**self.format_dict)
        return displayed

#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

#----------------------------------------------------------------------------

def gen_interp_video(G, mp4: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, device=torch.device(device_str), preheat=False, **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(torch.float32).to(device)
    ws = G.mapping(z=zs, c=None, truncation_psi=psi)
    _ = G.synthesis(ws[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    print('preheat:', preheat)
    if preheat:
        # Pre-heat by generating a single dummy image
        w = torch.from_numpy(interp(0)).to(torch.float32).to(device)
        img = G.synthesis(ws=w.unsqueeze(0), noise_mode='const')[0]

    # Render video.
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs) # when benchmarking with Intel VTune or NSight, this line (and the following ones referencing `video_out`) may need to be disabled, otherwise a "broken pipe" error may be encountered (related to passing the video frames to ffmpeg)
    for frame_idx in tqdm(range(num_keyframes * w_frames), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]"):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(torch.float32).to(device)
                img = G.synthesis(ws=w.unsqueeze(0), noise_mode='const')[0]
                imgs.append(img)
        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--output', help='Output .mp4 filename', type=str, required=True, metavar='FILE')
@click.option('--preheat', help='Generate a dummy frame before rendering the video', type=bool, default=False)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    output: str,
    preheat: bool
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device(device_str)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    G = try_ipex_optimize(G)


    #from torch.profiler import profile, record_function, ProfilerActivity
    #print('dir(ProfilerActivity)', dir(ProfilerActivity))
    #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #    with record_function("model_inference"):
    #with profile(activities=[
    #    ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True) as prof:
    #    with record_function("model_inference"):
    gen_interp_video(G=G, mp4=output, bitrate='12M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi)
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=10))
    #import time
    #print('sleeping')
    #time.sleep(3)
    global rates
    rates = np.array(rates)
    print('min/mean/median/max rate:', np.min(rates), np.mean(rates), np.median(rates), np.max(rates))
    #print('end')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
