import os.path
import typing
from typing import Dict, Tuple
import numpy as np
import torch
from geomloss import SamplesLoss
from matplotlib import pyplot as plt
from loguru import logger
import plotly.graph_objects as go
from torch.nn import Module

COLORS = ["blue", "red", "orange", "cyan", "magenta"]
# 3d plot angles https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
ELEVATION = 1.0
AZIMUTH = -80


def __plot_original_prior_posterior_2d(X0: torch.Tensor, X1: torch.Tensor, output_path: str,
                                       limits: Tuple[float, float]):
    plt.figure(figsize=(4, 4))
    plt.title(r"Samples from $\pi_0$ and $\pi_1$")
    plt.scatter(
        X0[:, 0].cpu().numpy(),
        X1[:, 1].cpu().numpy(),
        alpha=0.1,
        label=r"$\pi_0$",
        rasterized=True,
    )

    plt.scatter(
        X1[:, 0].cpu().numpy(),
        X1[:, 1].cpu().numpy(),
        alpha=0.1,
        label=r"$\pi_1$",
        rasterized=True,
    )
    plt.legend()
    plt.xlim(*limits)
    plt.ylim(*limits)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    # plt.show()


def __plot_original_prior_posterior_3d(X0: torch.Tensor, X1: torch.Tensor, output_path: str):
    X0_np = X0.detach().cpu().numpy()
    X1_np = X1.detach().cpu().numpy()
    fig = plt.figure()
    ax0 = fig.add_subplot(121, projection="3d", elev=ELEVATION, azim=AZIMUTH)
    ax1 = fig.add_subplot(122, projection="3d", elev=ELEVATION, azim=AZIMUTH)
    ax0.set_title(r"$\pi_0$")
    ax1.set_title(r"$\pi_1$")
    ax0.scatter(X0_np[:, 0], X0_np[:, 1], X0_np[:, 2], c=COLORS[0])
    ax1.scatter(X1_np[:, 0], X1_np[:, 1], X1_np[:, 2], c=COLORS[1])
    plt.savefig(output_path)
    # use plotly


def plot_samples_2d(X: torch.Tensor, output_path: str, limits: Tuple[float, float]):
    plt.figure(figsize=(4, 4))
    plt.title(r"Samples from $\pi_0$ and $T_\sharp \pi_1$")
    plt.scatter(
        X[:, 0].cpu().numpy(),
        X[:, 1].cpu().numpy(),
        alpha=0.1,
        label=r"$\pi_0$",
        rasterized=True,
    )
    plt.scatter(
        X[:, 0].cpu().numpy(),
        X[:, 1].cpu().numpy(),
        alpha=0.1,
        label=r"$T_\sharp \pi_1$",
        rasterized=True,
    )
    plt.legend()
    plt.xlim(*limits)
    plt.ylim(*limits)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    # plt.show()


def plot_samples_3d(X: torch.Tensor, output_path: str, limits: Tuple[float, float]):
    X_np = X.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", elev=ELEVATION, azim=AZIMUTH)
    ax.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2])
    plt.legend()
    plt.xlim(*limits)
    plt.ylim(*limits)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    # plt.show()
    # Use plotly
    fig = go.Figure(data=[go.Scatter3d(x=X_np[:, 0], y=X_np[:, 1], z=X_np[:, 2], mode='markers')])
    fig.update_layout(title='something')  # https://stackoverflow.com/a/63692534/5937273
    # fig.show()
    out_path_without_ext = output_path.split(".")[0]
    html_outpath = f"{out_path_without_ext}.html"
    fig.write_html(html_outpath)


def plot_samples(X: torch.Tensor, output_path: str, limits: Tuple[float, float]):
    D = X.shape[1]
    if D == 2:
        plot_samples_2d(X=X, output_path=output_path, limits=limits)
    elif D == 3:
        plot_samples_3d(X=X, output_path=output_path, limits=limits)
    else:
        raise ValueError(f"Unsupported plot with D = {D}")


def plot_original_prior_posterior(X0: torch.Tensor, X1: torch.Tensor, output_path: str, limits: Tuple[float, float]):
    """

    @param limits:
    @param output_path:
    @param X0:
    @param X1:
    @return:
    """
    assert len(list(X0.shape)) == len(list(X1.shape))
    nDims = len(list(X0.shape))
    for d_index in range(nDims):
        assert X0.shape[d_index] == X1.shape[d_index]
    # N = X0.shape[0]
    D = X0.shape[1]
    if D == 2:
        __plot_original_prior_posterior_2d(X0=X0, X1=X1, output_path=output_path, limits=limits)
    elif D == 3:
        __plot_original_prior_posterior_3d(X0=X0, X1=X1, output_path=output_path)
    else:
        raise ValueError(f"No support for plotting with D = {D}")


def plot_results(generation_results: Dict[str, Dict], x1: torch.Tensor, output_path: str, x_lim: typing.Tuple,
                 y_lim: typing.Tuple, samples_loss: Module) -> None:
    """
    This function compare models by
    1. Calculate Sinkhorn Values
    2. Plotting generated samples
    3. Plot Trajectories
    generation_results : dict[model_name,generated_samples]
    x1 : reference tensor of shape N X D
    z1 : generated tensor of shape N_ X D
    if N_ != N we down sample x1
    """
    # Calculate Sinkhorn Values
    assert len(x1.shape) == 2, "x1 (ref sample) original must be 2D : N X D"
    D = x1.shape[1]
    assert 2 <= D <= 3, "Supporting 2D and 3D plots only"
    n_subplots = len(generation_results) + 1
    plt.clf()  # clear any plots in buffer
    if D == 2:
        fig, axes = plt.subplots(nrows=1, ncols=n_subplots, figsize=(12, 6))
    elif D == 3:
        axes = []
        fig = plt.figure(figsize=(15, 10))
        for i in range(n_subplots):
            loc_index = 100 + (n_subplots * 10) + (i + 1)
            axes.append(fig.add_subplot(loc_index, projection="3d", elev=ELEVATION, azim=AZIMUTH))
    else:
        raise ValueError(f"Plotting for D = {D} is not supported")
    z_origin_np = x1.detach().cpu().numpy()
    if D == 2:
        axes[0].scatter(z_origin_np[:, 0], (z_origin_np[:, 1]), color=COLORS[0])
        axes[0].set_title(r"$\pi_1$")
        axes[0].set_xlim(x_lim)
        axes[0].set_ylim(y_lim)
    elif D == 3:
        axes[0].scatter(z_origin_np[:, 0], (z_origin_np[:, 1]), (z_origin_np[:, 2]), color=COLORS[0])
        axes[0].set_title(r"$\pi_1$")
        axes[0].set_xlim(x_lim)
        axes[0].set_ylim(y_lim)
        axes[0].set_zlim(y_lim)  # fixme
    else:
        raise ValueError(f"Plotting for D = {D} is not supported")
    for i, item in enumerate(generation_results.items()):
        model_name = item[0]
        z1 = item[1]["z1"]
        assert len(z1.shape) == 2, f"""z1 for {model_name}"""
        odesolve_method = item[1]["odesolve_method"]
        # calculate sinkhorn values
        assert z1.shape[1] == x1.shape[1]  # Assert z1 and x1 have the same dimension
        if z1.shape[0] < x1.shape[1]:
            x1 = x1[torch.randperm(n=z1.shape[0]), :]
        eval_dtype = torch.cuda.DoubleTensor # FIXME : need to make it parametric or global constant
        loss_value = samples_loss(z1.type(eval_dtype), x1.type(eval_dtype)).item()
        logger.info(f"Loss value for model : {model_name} = {loss_value}")
        if D == 2:
            axes[i + 1].set_xlim(x_lim)
            axes[i + 1].set_ylim(y_lim)
            axes[i + 1].scatter(z1[:, 0].detach().cpu().numpy(), z1[:, 1].detach().cpu().numpy(), color=COLORS[i + 1])
            axes[i + 1].set_title(f"model = {model_name}\n"
                                  f"odesolve_method = {odesolve_method}\n"
                                  f"sinkhorn(z1,x1) ={np.round(loss_value, 4)}")
        elif D == 3:
            axes[i + 1].set_xlim(x_lim)
            axes[i + 1].set_ylim(y_lim)
            axes[i + 1].set_zlim(y_lim)
            axes[i + 1].scatter(z1[:, 0], z1[:, 1], z1[:, 2], color=COLORS[i + 1])
            axes[i + 1].set_title(f"model = {model_name}\n"
                                  f"odesolve_method = {odesolve_method}\n"
                                  f"sinkhorn(z1,x1) ={np.round(loss_value, 4)}")
            fig.tight_layout()
    plt.savefig(output_path)
