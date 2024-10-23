from gpytorch.kernels import RBFKernel
from gpytorch.priors import Prior
from ot.backend import torch

if __name__ == "__main__":
    # u = RBFKernel()
    # u._set_lengthscale(value=torch.tensor([1.0]))
    # x1 = torch.tensor([1., 2., 3.])
    # x2 = torch.tensor([2., 3., 8.])
    # k = u(x1=x1, x2=x2)
    x1 = torch.tensor([1, 2, 3.0]).view(-1, 1)
    x2 = torch.tensor([2.0, 3.0]).view(1, -1)
    u = x1 - x2
    print(u)
