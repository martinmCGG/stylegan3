# can be run as "python test_kernels.py bias_act filtered_lrelu" to test the listed kernels, or just "python test_kernels.py" to test all.

import sys
import torch
import time

kernels_to_test = sys.argv[1:]
if len(kernels_to_test) == 0:
    kernels_to_test = ['bias_act', 'filtered_lrelu', 'upfirdn2d']

d='xpu'

for k in kernels_to_test:
    print('testing', k, flush=True)
    time.sleep(1)

    if k == 'bias_act':
        from torch_utils.ops import bias_act

        length = 100000000 # 100M

        for i in range(10):
            print(bias_act.bias_act(x=torch.ones([1,length], device=d), b=torch.ones([length], device=d), act='sigmoid'))

    elif k == 'upfirdn2d':
        from torch_utils.ops import upfirdn2d
        print(upfirdn2d.upfirdn2d(x=torch.ones([1,1,1,1], device=d), f=torch.ones([1,1], device=d)))

        #raise NotImplementedError('TODO')

#        """
    elif k == 'filtered_lrelu':
        from torch_utils.ops import filtered_lrelu
        print(filtered_lrelu.filtered_lrelu(x=torch.zeros([1,1,1,1]).to('xpu')))
        '''
The actual sizes ran during stylegan3-r 512x512 inference:
_filtered_lrelu_xpu(up= 2 , down= 2 , padding= [11, 10, 11, 10] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 1024, 36, 36]) , fu= torch.Size([12]) , fd= torch.Size([12, 12]) , b= torch.Size([1024]) , None, 0, 0)
_filtered_lrelu_xpu(up= 2 , down= 2 , padding= [11, 10, 11, 10] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 1024, 36, 36]) , fu= torch.Size([12]) , fd= torch.Size([12, 12]) , b= torch.Size([1024]) , None, 0, 0)
_filtered_lrelu_xpu(up= 4 , down= 2 , padding= [-2, -5, -2, -5] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 1024, 36, 36]) , fu= torch.Size([24]) , fd= torch.Size([12, 12]) , b= torch.Size([1024]) , None, 0, 0)
_filtered_lrelu_xpu(up= 2 , down= 2 , padding= [11, 10, 11, 10] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 1024, 52, 52]) , fu= torch.Size([12]) , fd= torch.Size([12, 12]) , b= torch.Size([1024]) , None, 0, 0)
_filtered_lrelu_xpu(up= 4 , down= 2 , padding= [-2, -5, -2, -5] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 1024, 52, 52]) , fu= torch.Size([24]) , fd= torch.Size([12, 12]) , b= torch.Size([1024]) , None, 0, 0)
_filtered_lrelu_xpu(up= 2 , down= 2 , padding= [11, 10, 11, 10] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 1024, 84, 84]) , fu= torch.Size([12]) , fd= torch.Size([12, 12]) , b= torch.Size([1024]) , None, 0, 0)
_filtered_lrelu_xpu(up= 4 , down= 2 , padding= [-2, -5, -2, -5] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 1024, 84, 84]) , fu= torch.Size([24]) , fd= torch.Size([12, 12]) , b= torch.Size([1024]) , None, 0, 0)
_filtered_lrelu_xpu(up= 2 , down= 2 , padding= [11, 10, 11, 10] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 967, 148, 148]), fu= torch.Size([12]) , fd= torch.Size([12, 12]) , b= torch.Size([967]) , None, 0, 0)
_filtered_lrelu_xpu(up= 4 , down= 2 , padding= [-2, -5, -2, -5] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 645, 148, 148]), fu= torch.Size([24]) , fd= torch.Size([12, 12]) , b= torch.Size([645]) , None, 0, 0)
_filtered_lrelu_xpu(up= 2 , down= 2 , padding= [11, 10, 11, 10] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 431, 276, 276]), fu= torch.Size([12]) , fd= torch.Size([12, 12]) , b= torch.Size([431]) , None, 0, 0)
_filtered_lrelu_xpu(up= 4 , down= 2 , padding= [-2, -5, -2, -5] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 287, 276, 276]), fu= torch.Size([24]) , fd= torch.Size([12, 12]) , b= torch.Size([287]) , None, 0, 0)
_filtered_lrelu_xpu(up= 2 , down= 2 , padding= [11, 10, 11, 10] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 192, 532, 532]), fu= torch.Size([12]) , fd= torch.Size([12, 12]) , b= torch.Size([192]) , None, 0, 0)
_filtered_lrelu_xpu(up= 2 , down= 2 , padding= [11, 10, 11, 10] , gain= 1.4142135623730951 , slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 128, 532, 532]), fu= torch.Size([12]) , fd= torch.Size([12]) , b= torch.Size([128]) , None, 0, 0)
_filtered_lrelu_xpu(up= 2 , down= 2 , padding= [-9, -10, -9, -10], gain= 1.4142135623730951, slope= 0.2 , clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 128, 532, 532]), fu= torch.Size([12]) , fd= torch.Size([12]) , b= torch.Size([128]) , None, 0, 0)
_filtered_lrelu_xpu(up= 1 , down= 1 , padding= [0, 0, 0, 0] ,     gain= 1 ,                  slope= 1 ,   clamp= 256 , flip_filter= False ).apply(x= torch.Size([1, 3, 512, 512]) ,  fu= None ,             fd= None ,             b= torch.Size([3]) , None, 0, 0)
        '''
        print(filtered_lrelu.filtered_lrelu(up= 2 , down= 2 , padding= [-9, -10, -9, -10], gain= 1.4142135623730951, slope= 0.2 , clamp= 256 , flip_filter= False, x= torch.ones([1, 128, 532, 532],device=d), fu= torch.ones([12],device=d) , fd= torch.ones([12],device=d) , b= torch.ones([128],device=d)))
#"""
    else:
        print('unknown kernel', k)
        exit(1)
    
    time.sleep(1)

print('done')