import torch
from torch import Tensor
import numpy as np

def _is_integer(x) -> bool:
    r"""Type check the input number is an integer.
    Will return True for int, SymInt and Tensors with integer elements.
    """
    if isinstance(x, (int, torch.SymInt, np.integer)): # HACK: also check for np.integer because numpy.int64 is not a subclass of int
        return True
    return isinstance(x, Tensor) and not x.is_floating_point()

torch.nn.functional._is_integer = _is_integer