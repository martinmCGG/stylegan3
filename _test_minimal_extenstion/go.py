#sudo docker run --rm -it --privileged --device=/dev/dri --ipc=host -v /home/user/stylegan3/_test_minimal_extenstion/:/test -w /test -u 0:0 ipex bash -c '. /opt/intel/oneapi/setvars.sh; python go.py'

import torch
import intel_extension_for_pytorch
from torch.xpu.cpp_extension import load

lltm_xpu = load(name="lltm_xpu", sources=['lltm_xpu.cpp', 'lltm_xpu_kernel.cpp',])

print(lltm_xpu)



# adapted from https://github.com/pytorch/extension-cpp/blob/master/check.py
batch_size = 3
features = 17
state_size = 5
kwargs = {'dtype': torch.float32,
          'device': torch.device("xpu"),
          'requires_grad': True}

X = torch.randn(batch_size,
                features,
                **kwargs)
h = torch.randn(batch_size, state_size, **kwargs)
C = torch.randn(batch_size, state_size, **kwargs)
W = torch.randn(3 * state_size, features + state_size, **kwargs)
b = torch.randn(1, 3 * state_size, **kwargs)

outputs = lltm_xpu.forward(X, W, b, h, C)

print("outputs", outputs)
