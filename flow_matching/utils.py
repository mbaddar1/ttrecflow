import functools
from contextlib import contextmanager
from timeit import default_timer
from typing import Callable, Literal, Optional, List

import ot
import pandas as pd
import seaborn as sns
import torch
from jaxtyping import Float

from flow_matching.tt.basis import Basis

Array = torch.Tensor

def relative_l2(
        y_true: Float[Array, "B ..."],
        y_hat: Float[Array, "B ..."],
        weights: Optional[Float[Array, "B"]] = None,
) -> float:
    """Calculate the relative L2 error between two arrays.

    This function supports weighting of the error by a third array.

    Parameters
    ----------
    y_true : Float[Array, "B ..."]
        The ground truth array. The shape of this array should start with a batch
        dimension "B", followed by any number of additional dimensions.
    y_hat : Float[Array, "B ..."]
        The predicted array. The shape of this array should match that of `y_true`.
    weights : Optional[Float[Array, "B"]], optional
        A array of weights for each example in the batch. If provided, the L2 error
        for each example will be multiplied by the corresponding weight before being
        summed. If not provided, defaults to an array of ones.


    Returns
    -------
    float
        The relative L2 error between `y_true` and `y_hat`.

    Raises
    ------
    ValueError
        If the first dimensions of `y_true` and `y_hat` do not match.
    """
    if y_true.shape[0] != y_hat.shape[0]:
        raise ValueError("The first dimension should be the same for both tensors.")

    if weights is None:
        weights = torch.ones(y_true.shape[0])
    abs_l2 = float(torch.sqrt(torch.sum(weights * (y_true - y_hat) ** 2)).item())
    norm = float(torch.sqrt(torch.sum(weights * y_true ** 2)).item())
    return abs_l2 / norm


@contextmanager
def elapsed_timer():
    t1 = t2 = default_timer()
    yield lambda: t2 - t1
    t2 = default_timer()


def plot_marginals(
        data: Float[Array, "B d"],
        *,
        diag_kind: Literal["auto", "hist", "kde"] = "kde",
        contour_lower: bool = False,
) -> sns.PairGrid:
    df = pd.DataFrame(data, columns=[f"dim_{i + 1}" for i in range(data.shape[1])])
    g = sns.pairplot(data=df, diag_kind=diag_kind, corner=True)
    g.set(xlabel=None, ylabel=None)
    if contour_lower:
        g.map_lower(sns.kdeplot)
    return g


@torch.no_grad()
def non_uniform_grid(
        inv_cdf: Callable[[Float[Array, "grid_size"]], Float[Array, "grid_size"]],
        grid_size: int,
        p: float = 1e-8,
) -> Float[Array, "grid_size"]:
    """Create a non-uniform grid based on the inverse cumulative distribution function (CDF).

    This function generates a non-uniform grid of size `grid_size` using the inverse cumulative
    distribution function `inv_cdf`. The parameter `p` is the probability of being smaller than `p`.
    The non-transformed generated grid is thus defined on :math:`[p,1-p]`, and then
    is applied the inverse CDF.
    This results in a non-uniform grid that is more dense in regions where
    the probability density function (PDF) is higher, and less dense in regions where the PDF is
    lower.

    Parameters
    ----------
    inv_cdf : Callable[[Float[Array, "grid_size"]], Float[Array, "grid_size"]]
        The inverse cumulative distribution function. This function takes an array of values
        between 0 and 1, and returns an array of the same shape containing the corresponding
        values from the distribution.
    grid_size : int
        The number of points in the grid.
    p : float, optional
        The lower and upper bounds of the uniform grid, by default 1e-8. The uniform grid is
        generated between `p` and `1 - p`. This is done to avoid numerical issues when computing
        the inverse CDF at values very close to 0 or 1.

    Returns
    -------
    Float[Array, "grid_size"]
        The non-uniform grid of size `grid_size`.
    """
    grid = torch.linspace(p, 1.0 - p, steps=grid_size)
    return inv_cdf(grid)


# TODO: Wasserstein distances
@torch.no_grad()
def wasserstein_distance(
        x0: Float[Array, "n_samples d"],
        x1: Float[Array, "n_samples d"],
        entropic_reg: float = 1e-3,
        power: int = 2,
) -> float:
    if not isinstance(entropic_reg, float) or entropic_reg < 0.0:
        raise ValueError("'entropic_reg' must be >= 0.")
    if power < 1:
        raise ValueError("'power' must be >= 1.")

    if entropic_reg == 0.0:
        ot_fn = ot.emd2
    else:
        ot_fn = functools.partial(ot.sinkhorn2, reg=entropic_reg)

    a, b = ot.unif(x0.shape[0]), ot.unif(x1.shape[0])
    M = torch.cdist(x0, x1, p=2)
    M = M ** power
    ret = ot_fn(
        a=a, b=b, M=M.cpu().numpy(), numItermax=int(1e7), log=False, return_matrix=False
    )
    assert isinstance(ret, float)
    return ret ** (1 / power)


def f_divergence(
        P: Float[Array, "n_samples"],
        Q: Float[Array, "n_samples"],
        f: Callable[[Float[Array, "N"]], Float[Array, "N"]],
) -> float:
    y = f(P / Q)
    return float(torch.sum(y * Q).item())


total_variation = functools.partial(f_divergence, f=lambda x: 0.5 * abs(x - 1))
kl_divergence = functools.partial(f_divergence, f=lambda x: x * torch.log(x))
hellinger_distance = functools.partial(
    f_divergence, f=lambda x: 0.5 * (torch.sqrt(x) - 1.0) ** 2
)
chi2_divergence = functools.partial(f_divergence, f=lambda x: (x - 1.0) ** 2)
