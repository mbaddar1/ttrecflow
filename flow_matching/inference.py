from typing import Optional

import torch
from jaxtyping import Float
from torchdiffeq import odeint

from flow_matching.tt import FTT

Array = torch.Tensor

__all__ = ["infer"]


@torch.no_grad()
def infer(
    vector_field: list[FTT],
    X0: Float[Array, "B d"],
    t_eval: Float[Array, "time_steps"],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: Optional[str] = None,
) -> Float[Array, "time_steps B d"]:
    B, d = X0.shape

    def _f(t: float, Y: Float[Array, "B*d"]) -> Float[Array, "B*d"]:
        Y = Y.reshape(B, d)

        inputs = torch.cat([Y, torch.ones(B, 1).type(X0.dtype).to(X0.device) * t], dim=1)
        result = torch.zeros(B, d).type(X0.dtype).to(X0.device)
        for i, ftt in enumerate(vector_field):
            result[:, i] = ftt(inputs).ravel()
        return result.ravel()

    sol: Float[Array, "time_steps*B*d"] = odeint(
        func=_f, y0=X0.ravel(), t=t_eval, rtol=rtol, atol=atol, method=method
    )
    sol_reshaped = sol.reshape(-1, B, d)
    return sol_reshaped
