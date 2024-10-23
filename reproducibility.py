import random
import numpy as np
import torch
from numpy.random import Generator


## PyTorch Reproducibility ##
# https://pytorch.org/docs/stable/notes/randomness.html
# https://gist.github.com/Guitaricet/28fbb2a753b1bb888ef0b2731c03c031


def set_env(seed: int)->None:
    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # FIXME, rethink default device and dtype : now setting them explicitly
    # torch.backends.cudnn.benchmark = False
    # torch.set_default_dtype(default_dtype)

