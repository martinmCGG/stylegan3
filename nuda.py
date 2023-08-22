# NUDA: "Non-proprietary Unified Device Architecture", a replacemnet for CUDA-specific things

import torch
import time

class Event:
    def __init__(self, enable_timing=False) -> None:
        self.enable_timing = enable_timing

    def record(self, stream):
        self.starttime = time.process_time()

    def synchronize(self):
        self.endtime = time.process_time()

    def elapsed_time(self, end_event):
        return self.endtime - self.starttime

torch.cuda.Event = Event


def return_self(self, device=None):
    return self

torch.Tensor.pin_memory = return_self

def return_dummy(arg):
    return 'dummy'

torch.cuda.current_stream = return_dummy
