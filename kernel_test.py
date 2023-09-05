import torch
import intel_extension_for_pytorch
import torch_utils.ops.bias_act as ba
import random

sizes = [
(torch.Size([1, 512]), torch.Size([512])),
(torch.Size([1, 512]), torch.Size([512])),
(torch.Size([1, 1024, 36, 36]), torch.Size([1024])),
(torch.Size([1, 1024, 82, 82]), None),
(torch.Size([1, 1024, 36, 36]), torch.Size([1024])),
(torch.Size([1, 1024, 82, 82]), None),
(torch.Size([1, 1024, 36, 36]), torch.Size([1024])),
(torch.Size([1, 1024, 114, 114]), None),
(torch.Size([1, 1024, 52, 52]), torch.Size([1024])),
(torch.Size([1, 1024, 114, 114]), None),
(torch.Size([1, 1024, 52, 52]), torch.Size([1024])),
(torch.Size([1, 1024, 178, 178]), None),
(torch.Size([1, 1024, 84, 84]), torch.Size([1024])),
(torch.Size([1, 1024, 178, 178]), None),
(torch.Size([1, 1024, 84, 84]), torch.Size([1024])),
(torch.Size([1, 1024, 306, 306]), None),
(torch.Size([1, 967, 148, 148]), torch.Size([967])),
(torch.Size([1, 967, 306, 306]), None),
(torch.Size([1, 645, 148, 148]), torch.Size([645])),
(torch.Size([1, 645, 562, 562]), None),
(torch.Size([1, 431, 276, 276]), torch.Size([431])),
(torch.Size([1, 431, 562, 562]), None),
(torch.Size([1, 287, 276, 276]), torch.Size([287])),
(torch.Size([1, 287, 1074, 1074]), None),
(torch.Size([1, 192, 532, 532]), torch.Size([192])),
(torch.Size([1, 192, 1074, 1074]), None),
(torch.Size([1, 128, 532, 532]), torch.Size([128])),
(torch.Size([1, 128, 1074, 1074]), None),
(torch.Size([1, 128, 532, 532]), torch.Size([128])),
(torch.Size([1, 128, 1034, 1034]), None),
(torch.Size([1, 3, 512, 512]), torch.Size([3])),
(torch.Size([1, 3, 512, 512]), None),
]

for x_size, b_size in sizes:
    x = torch.randn(x_size)
    b = None if b_size is None else torch.randn(b_size)
    #print(x_size, b_size)
    #print(x.shape, b.shape)

    act=random.choice(list(ba.activation_funcs.keys()))

    y_ref = ba.bias_act(x, b, dim=1, act=act, impl='ref')
    y_xpu = ba.bias_act(x.to('xpu'), b.to('xpu') if b is not None else None, dim=1, act=act, impl='xpu')

    diff = torch.mean(torch.abs((y_ref - y_xpu.cpu()).flatten()))
    print(diff, 'diff')
    assert diff < 1e-7
    