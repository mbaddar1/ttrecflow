from typing import Optional

import torch
from jaxtyping import Float

from flow_matching.flow import Flow
from flow_matching.tt import FTT, Rule, als_tt

Array = torch.Tensor

__all__ = ["learn_flow_matching"]


@torch.no_grad()
def learn_flow_matching(
    X0: Float[Array, "B d"],
    X1: Float[Array, "B d"],
    t: Float[Array, "B"],
    flow: Flow,
    initial_guess: list[FTT],
    *,
    rank_adaptivity: Optional[Rule] = None,
    regularisation: Optional[float] = None,
    max_iters: int = 100,
    tolerance: float = 1e-7,
    verbose: int = 0,
) -> list[FTT]:
    if X0.ndim != 2:
        raise ValueError("'X0' should be a 2D array.")
    if X1.ndim != 2:
        raise ValueError("'X1' should be a 2D array.")
    if X0.shape != X1.shape:
        raise ValueError("'X0' and 'X1' must be of the same shape.")
    if len(initial_guess) != X0.shape[1]:
        raise ValueError(
            "There should be the same number of FTTs as the number of dimensions."
        )

    X_t = flow(X0=X0, t=t, X1=X1)  # (B, d)
    inputs = torch.cat([X_t, t], dim=1)  # (B, d+1)
    outputs = flow.derivative(X0=X0, t=t, X1=X1)  # (B, d)

    learned_ftts = [None] * len(initial_guess)
    for i, guess in enumerate(initial_guess):
        features = [
            basis(inputs[:, k]) for k, basis in enumerate(guess._bases)
        ]  # (B, m_k)

        iters, stag, result = als_tt(
            guess=guess._tt,
            A=features,
            b=outputs[:, i],
            max_iters=max_iters,
            stagnation=tolerance,
            l2_regularization=regularisation,
            rank_adaptivity=rank_adaptivity,
            verbose=verbose,
        )
        result.set_core(result.order - 1)
        learned_ftts[i] = FTT(result, guess._bases)

    return learned_ftts
