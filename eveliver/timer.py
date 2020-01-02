import collections
import time

import torch


class PyTorchTimer:
    def __init__(self):
        self.ac = collections.defaultdict(int)
        self.t = None

    def tick(self, name):
        torch.cuda.synchronize()
        current_time = time.time()
        if name is not None:
            assert self.t is not None
            self.ac[name] += (current_time - self.t)
        self.t = current_time

    def __repr__(self):
        return self.ac.__repr__()
