from typing import List
import torch
from geomloss import SamplesLoss
from torch.nn import Module

from flow_matching import infer
from flow_matching.tt import FTT
from loguru import logger

ODESOLVE_METHODS = ["rk4", "euler"]


def eval_ftt(model: List[FTT], X0: torch.Tensor, X1: torch.Tensor, N: int, ode_solve_method: str,
             Ainv: torch.Tensor, b: torch.Tensor, samples_loss: SamplesLoss,
             **kwargs) -> float:
    """
    @param samples_loss:
    @param model: TT-RBF Model
    @param X0: The initial sample
    @param X1: The reference target sample
    @param N: Number of time points
    @param ode_solve_method : method of ode_solve
    @param Ainv : for linear transformation
    @param b : for linear transformation
    @return: sinkhorn value between generated sample X^hat_1 and reference sample X1
    """
    # check input
    assert isinstance(samples_loss, SamplesLoss)
    # infer
    eval_dtype = torch.cuda.DoubleTensor
    t = torch.linspace(0.0, 1.0, steps=N)
    logger.info(f"Running inference with {N} time steps with ode_solve_method = {ode_solve_method}")
    traj = infer(vector_field=model, X0=X0, t_eval=t, method=ode_solve_method)
    X1_hat = (traj[-1, :, :] - b) @ Ainv.T
    # eval
    samples_loss_print_str = (f"SamplesLoss(loss={samples_loss.loss},scaling={samples_loss.scaling},"
                              f"p={samples_loss.p},blur={samples_loss.blur})")
    logger.info(f"Evaluating generated samples using {samples_loss_print_str}")
    loss_value = samples_loss(X1_hat.type(eval_dtype), X1.type(eval_dtype)).item()
    return loss_value
