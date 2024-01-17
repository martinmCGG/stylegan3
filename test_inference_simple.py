import torch
import intel_extension_for_pytorch as ipex
import dnnlib
import numpy as np

import PIL.Image

import legacy
device = torch.device('xpu')

#network_pkl='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl'
network_pkl='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl'
#network_pkl='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl'

with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

z = torch.from_numpy(np.random.RandomState(seed=2).randn(1, G.z_dim)).to(torch.float32).to(device)

img = G(z, None)

print(img.shape, img.device)

img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'test_inference.png')
