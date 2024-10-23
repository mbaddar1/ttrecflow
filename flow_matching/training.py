from typing import List

import torch
from datetime import datetime
from flow_matching import learn_flow_matching, LinearInterpolation
from flow_matching.tt import FTT, TT
from flow_matching.tt.basis import BSplines, RBFPyTorchWrapper
import numpy as np
from loguru import logger
from tqdm import tqdm

rng = np.random.default_rng(0)


def get_train_tuple(z0=None, z1=None):
    t = torch.rand((z1.shape[0], 1)).type(z0.dtype).to(z0.device)
    z_t = t * z1 + (1. - t) * z0
    target = z1 - z0
    return z_t, t, target


def train_rectified_flow_nn(rectified_flow_nn, optimizer, pairs, batchsize, inner_iters):
    loss_curve = []
    alpha = 0.1
    loss_fn = torch.nn.MSELoss()
    si = None
    for i in tqdm(range(inner_iters + 1), desc="training recflow-nn "):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = get_train_tuple(z0=z0, z1=z1)

        pred = rectified_flow_nn.model(z_t, t)
        residuals = loss_fn(pred, target)
        # norm_ = rectified_flow_nn.model.norm()
        # loss = residuals + reg_coeff * norm_
        # both losses are the same
        # loss = 0.5 * (target - pred).view(pred.shape[0], -1).pow(2).sum(dim=1)
        # loss = loss.mean()
        if si is None:
            si = residuals.item()
        else:
            si = alpha * residuals.item() + (1 - alpha) * si
        if i % 100 == 0:
            logger.info(f"si for loss type {type(loss_fn)} @i = {i} => {si}")

        residuals.backward()

        optimizer.step()
        loss_curve.append(np.log(residuals.item()))  # to store the loss curve

    return rectified_flow_nn, loss_curve


def train_tt_recflow(rank: int, bases_model: str, iterations: int, reg_coeff: float, tol: float, verbose: int,
                     X0: torch.Tensor,
                     X1: torch.Tensor, **kwargs):
    """
    @param rank:
    @param bases_model:
    @param iterations:
    @param reg_coeff:
    @param tol:
    @param verbose:
    @param X0:
    @param X1:
    @param kwargs:
    @return:
    """
    assert (len(X0.shape)) == 2, "X0 must be N X D tensor"
    assert len(list(X1.shape)) == 2, "X1 must be N X D tensor"
    for i in range(2):
        assert X0.shape[i] == X1.shape[i], f"dim # {i} of X0 = {X0.shape[i]} != dim # {i} of X1 = {X1.shape[i]}"
    n_samples_train = X0.shape[0]
    D = X0.shape[1]
    initial_ranks = [1] + [rank] * D + [1]
    # t = torch.rand(n_samples, 1)
    # t = torch.from_numpy(rng.random((n_samples_train, 1)))
    t = torch.distributions.Uniform(low=0, high=1).sample(sample_shape=torch.Size([n_samples_train, 1])).type(
        X0.dtype).to(X0.device)
    # domains = [limits for _ in range(D_)] + [[0.0, 1.0]] # TODO what is it and where it is used ??
    if bases_model == "bsplines":
        logger.info(f"Creating Bsplines Bases")
        bspline_degree = kwargs["degree"]
        n_knots = kwargs["n_knots"]
        grid_size = 1 + bspline_degree + n_knots
        # grid = non_uniform_grid(norm.icdf, grid_size, p=1e-30)
        grid_ = torch.linspace(-10.0, 10.0, steps=grid_size)
        bases = [BSplines(grid_, bspline_degree) for i in range(D)] + [
            BSplines(torch.linspace(0.0, 1.0, steps=grid_size), bspline_degree)]
    elif bases_model == "rbf":
        bases = []
        n_centres = kwargs["n_centres"]
        length_scales = kwargs["length_scales"]
        assert len(
            length_scales) == D + 1, f"len(length_scales)must equal D+1 , where D={D} , {len(length_scales)}!={D + 1}"

        for i in range(D + 1):
            if i <= D - 1:  # for x_i
                X0_min = torch.min(X0[:, i]).item()
                X1_max = torch.max(X1[:, i]).item()
                centres = torch.linspace(start=X0_min, end=X1_max, steps=n_centres).type(X0.dtype).to(X0.device)
                # FIXME, we can pass explicit dtype
                bases.append(RBFPyTorchWrapper(centers=centres, length_scale=length_scales[i]))
            else:  # i = D Then add basis for t
                bases.append(
                    RBFPyTorchWrapper(
                        centers=torch.linspace(start=0.0, end=1.0, steps=n_centres).type(X0.dtype).to(X0.device),
                        length_scale=length_scales[i]))

    else:
        raise NotImplementedError(f"bases : {bases_model} is not supported, yet !")
    dims = [basis.dim for basis in bases]

    initial_guess = [FTT(TT.rand(dims, initial_ranks, X0.dtype, X0.device), bases) for _ in range(D)]
    # set cores device and dtype
    # for ftt_ in initial_guess:
    #     for i in range(len(ftt_._tt._cores)):
    #         ftt_._tt._cores[i] = ftt_._tt._cores[i].type(X0.dtype).to(X0.device)

    # # FIXME - for debugging - remove later
    # for i, init_guess_ in enumerate(initial_guess):
    #     logger.info(f"norm(init_guess[{i}])= {init_guess_.norm()}")
    # # FIXME - end of the debugging code - to remove later
    # # ALS parameters
    # # rule = DoerflerAdaptivity(delta=1e-8, max_ranks=[10] * d, dims=dims, verbose=True)
    rule = None

    logger.info(f"Basis dimensions: {dims}")
    logger.info(f"Initial ranks: {initial_ranks}")
    if reg_coeff:
        logger.info(f"Initial L2 regularisation: {reg_coeff:.2e}")
    else:
        logger.info("Initial L2 regularisation: None")

    logger.info(f"Max iterations: {iterations}")
    logger.info(f"Tolerance: {tol:.2e}")
    logger.info(f"Rank adaptivity: {rule}")

    logger.info("Learning the vector field")

    ftt_results_model: List[FTT] = []  # FIXME , remove later
    # for j in range(2):  # FIXME, just to test if repeated calls with same init_guess will lead to the same model
    start_time_ = datetime.now()
    ftt_results_model: List[FTT] = learn_flow_matching(
        X0=X0,
        X1=X1,
        t=t,
        flow=LinearInterpolation(T=1.0),
        initial_guess=initial_guess,
        rank_adaptivity=rule,
        regularisation=reg_coeff,
        max_iters=iterations,
        tolerance=tol,
        verbose=verbose,
    )
    end_time_ = datetime.now()
    logger.info(f"Training time for TT-Recflow = {(end_time_ - start_time_).seconds} seconds")
    # # FIXME for debugging only  - remove later
    # for ftt_idx, ftt_ in enumerate(ftt_results_model):
    #     logger.info(f"norm(fitted_ftt[{ftt_idx}]) @itr {j}= {ftt_.norm()}")
    # # FIXME - end the debugging code snippet- to remove later
    return ftt_results_model


def find_rbf_centers():
    """
    Clustering
    RBFN with clustering for centers
    https://github.com/Yangyangii/MachineLearningTutorial/blob/master/Numpy/RBF-Network-with-Kmeans-clustering.ipynb
    https://inria.hal.science/hal-01420235/document
    https://www.saedsayad.com/artificial_neural_network_rbf.htm#:~:text=Any%20clustering%20algorithm%20can%20be,centers%20of%20the%20RBF%20units.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9601673/
    @return:
    """
    pass
