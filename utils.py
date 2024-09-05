import torch

from torch import Tensor
from torch.nn import Sequential

class DeNormalize:
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
class Hook:
    def __init__(self, model_layer: Sequential, backward=False):
        if backward:
            self.hook = model_layer.register_backward_hook(self.hook_fn)
        else:
            self.hook = model_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, _, input: Tensor, output: Tensor):
        self.input = input
        self.output = output
    
    def close(self):
        self.hook.remove()