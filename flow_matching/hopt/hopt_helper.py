import pickle
import time
import torch
from geomloss import SamplesLoss
from hyperopt import STATUS_OK, hp, Trials, fmin, tpe
from typing import Tuple, List

from torch.nn import Module

from flow_matching.eval import eval_ftt
from flow_matching.training import train_tt_recflow
from loguru import logger

from flow_matching.tt import FTT


def do_hopt(X0_train: torch.Tensor, X0_test: torch.Tensor,
            X1_train: torch.Tensor, X1_test: torch.Tensor, N_time_steps: int, basis_model: str,
            tol: float, tt_als_iterations: int, verbose: int, ode_solve_method: str,
            max_evals: int,
            samples_loss: SamplesLoss) -> Tuple[List[FTT], dict, float]:
    space = {"l": hp.uniform("l", 1e-5, 2.0),
             "r": hp.randint("r", 2, 10),
             "reg_coeff": hp.uniform("reg_coeff", 1e-4, 1e-1),
             "nc": hp.randint("nc", 10, 40),
             "X0_train": X0_train,
             "X0_test": X0_test,
             "X1_train": X1_train,
             "X1_test": X1_test,
             "N_time_steps": N_time_steps,
             "bases_model": basis_model,
             "tol": tol,
             "iterations": tt_als_iterations,
             "ode_solve_method": ode_solve_method,
             "loss": samples_loss,
             "verbose": verbose}
    assert isinstance(samples_loss, SamplesLoss)
    trials = Trials()
    best_parameters = fmin(fn=hopt_tt_recflow_obj_fn, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    min_loss = trials.best_trial['result']['loss']
    best_model: List[FTT] = trials.best_trial["result"]["model"]
    assert isinstance(best_model, List)
    for ftt_ in best_model:
        assert isinstance(ftt_, FTT)
    logger.info(f"best params = {best_parameters}")
    return best_model, best_parameters, min_loss


def hopt_tt_recflow_obj_fn(args):
    X0_train = args["X0_train"]
    X1_train = args["X1_train"]
    X0_test = args["X0_test"]
    X1_test = args["X1_test"]
    loss = args["loss"]
    r = args["r"]
    D = X0_train.shape[1]
    n_centres = args["nc"]
    length_scale = args["l"]
    tt_recflow_ode_solve_method = args["ode_solve_method"]
    length_scales = [length_scale] * (D + 1)

    tol = args["tol"]
    iterations = args["iterations"]
    kwargs = {"length_scales": length_scales, "n_centres": n_centres}
    # Affine transformation
    sigma_X1 = torch.cov(X1_train.T)  # (d, d)
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma_X1)
    A = eigenvectors.T
    Ainv = torch.linalg.inv(A)
    b = -A @ torch.mean(X1_train, dim=0)
    X1_transformed = torch.mm(X1_train, A.T) + b
    logger.info(f"Running hyperopt-objective instance with l = {length_scales},r={r},n_centres = {n_centres}")
    ftt_results_model = train_tt_recflow(rank=r, bases_model=args["bases_model"],
                                         iterations=iterations, reg_coeff=args["reg_coeff"],
                                         tol=tol, X0=X0_train, X1=X1_transformed, verbose=args["verbose"], **kwargs)
    quality_metric_value = eval_ftt(model=ftt_results_model, X0=X0_test, X1=X1_test, N=args["N_time_steps"],
                                    ode_solve_method=tt_recflow_ode_solve_method, Ainv=Ainv, b=b, samples_loss=loss,
                                    **kwargs)
    logger.info(
        f"Hyperopt run  with r={r},l={length_scale},m = {n_centres} and quality_metric_value = {quality_metric_value}")
    # get the best model
    return {
        'loss': quality_metric_value,
        'model': ftt_results_model,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time()
        # -- attachments are handled differently
        # 'attachments':
        #     {'time_module': pickle.dumps(time.time),
        #      "model": ftt_results_model,
        #      "sinkhorn_value": sinkhorn_value}
    }
