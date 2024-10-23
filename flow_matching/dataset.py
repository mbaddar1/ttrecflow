from typing import Dict
import numpy as np
import torch
from numpy.random import Generator
from sklearn.datasets import make_s_curve, make_moons, make_circles, make_swiss_roll, make_blobs

SUPPORTED_DATASETS = ["moons-sklearn", "circles-sklearn", "swissroll-2d-sklearn", "swissroll", "moons", "circles",
                      "gaussian-mixture", "blobs-sklearn", "uniform-sphere", "s-curve-2d-sklearn",
                      "swissroll-3d-sklearn",
                      "s-curve-3d-sklearn"]
DATASETS_DIMENSIONS = {"moons-sklearn": 2, "circles-sklearn": 2, "swissroll-2d-sklearn": 2, "swissroll": 2, "moons": 2,
                       "circles": 2, "gaussian-mixture": 2, "blobs-sklearn": 2, "uniform-sphere": 2,
                       "s-curve-2d-sklearn": 2,
                       "swissroll-3d-sklearn": 3, "s-curve-3d-sklearn": 3}
# assert consistency between SUPPORTED_DATASETS and DATASETS_DIMENSIONS
assert len(SUPPORTED_DATASETS) == len(DATASETS_DIMENSIONS.keys())  # having same number of entries
for ds_name in SUPPORTED_DATASETS:
    assert ds_name in DATASETS_DIMENSIONS.keys(), f"dataset : {ds_name} has no pre-defined dimension"


def my_swissroll(n_samples: int):
    t = 1.5 * torch.pi * (1.0 + 2.0 * torch.distributions.Uniform().sample(sample_shape=torch.Size([n_samples])))
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    samples = (
            0.8
            * torch.from_numpy(
        torch.vstack((x, y)) + 0.5 * torch.distributions
        .MultivariateNormal(loc=torch.zeros(2),
                            covariance_matrix=torch.eye(2)).sample(
            sample_shape=torch.Size([2, n_samples]))
    ).T)
    return samples


def my_make_moons(n_samples: int, dim: int, rng):
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
    samples_1 = (
            dim
            * np.vstack(
        [
            np.append(outer_circ_x, inner_circ_x),
            np.append(outer_circ_y, inner_circ_y),
        ]
    ).T
    )
    samples_1 += 0.1 * rng.standard_normal(size=samples_1.shape)
    samples_1 = torch.from_numpy(samples_1)
    return samples_1


def my_gaussian_mixtures(n_samples: int, dim: int) -> torch.Tensor:
    var: float = 0.3
    n_comp: int = 8
    t = 2 * torch.pi * torch.arange(n_comp) / n_comp
    mu_target = dim * torch.stack([-np.sin(t), np.cos(t)])
    cov_target = var * torch.stack([torch.eye(2) for _ in range(n_comp)])
    indices = np.random.randint(low=0, high=n_comp)  # rng.integers(0, n_comp, size=(n_samples,))
    samples = torch.zeros((n_samples, dim))
    for i in range(n_comp):
        idx = indices == i
        samples[idx] = torch.distributions.MultivariateNormal(loc=mu_target, covariance_matrix=cov_target).sample(
            sample_shape=torch.Size([idx.sum()]))
    return samples


def generate_dataset(dataset_name: str, n_samples_train: int, n_samples_test: int, X0_train: torch.Tensor,
                     X0_test: torch.Tensor) -> Dict:
    data_dict = {}
    if dataset_name == "gaussian-mixture":
        D = 8.0  # 10.0
        x_lim = (-10, 10)
        y_lim = (-10, 10)
        data_dim = 2
        X1_train = my_gaussian_mixtures(n_samples=n_samples_train, dim=data_dim)
        X1_test = my_gaussian_mixtures(n_samples=n_samples_test, dim=data_dim)


    elif dataset_name == "circles-sklearn":
        D = 8.0
        noise_ = 0.005
        scale_ = 5.0
        factor = 0.3
        x_lim = (-10, 10)
        y_lim = (-10, 10)
        X1_train = torch.tensor(
            make_circles(n_samples=n_samples_train, shuffle=True, factor=factor, noise=noise_)[0] * scale_)
        X1_test = torch.tensor(
            make_circles(n_samples=n_samples_test, shuffle=True, factor=factor, noise=noise_)[0] * scale_)


    elif dataset_name == "swissroll":
        D = 8.0  # 10.0
        # target is swiss roll

        x_lim = (-12, 12)
        y_lim = (-12, 12)
        X1_train = my_swissroll(n_samples=n_samples_train)
        X1_test = my_swissroll(n_samples=n_samples_test)


    elif dataset_name == "swissroll-2d-sklearn":
        D = 8.0
        noise_ = 0.005
        scale_ = 2.0
        x_lim = (-10, 10)
        y_lim = (-10, 10)
        X1_train = torch.tensor(make_swiss_roll(n_samples=n_samples_train, noise=noise_)[0][:, [0, 2]] / scale_)
        X1_test = torch.tensor(make_swiss_roll(n_samples=n_samples_test, noise=noise_)[0][:, [0, 2]] / scale_)
    elif dataset_name == "uniform-sphere":
        D = 8.0  # 10.0
        x_lim = (-12, 12)
        y_lim = (-12, 12)
        X1_train = D * X0_train / torch.linalg.norm(X0_train, ord=2, dim=1, keepdim=True)
        X1_test = D * X0_test / torch.linalg.norm(X0_test, ord=2, dim=1, keepdim=True)
    elif dataset_name == "blobs-sklearn":
        D = 8.0
        x_lim = (-15, 15)
        y_lim = (-15, 15)
        X1_train = torch.tensor(make_blobs(n_samples=n_samples_train, n_features=2)[0])
        X1_test = torch.tensor(make_blobs(n_samples=n_samples_test, n_features=2)[0])
    elif dataset_name == "moons":
        D = 2.0
        x_lim = (-5, 5)
        y_lim = (-3, 3)
        X1_train = my_make_moons(n_samples=n_samples_train)
        X1_test = my_make_moons(n_samples=n_samples_test)

    elif dataset_name == "moons-sklearn":
        D = 2.0
        noise = 5e-3
        scale = 5.0
        x_lim = (-10, 15)
        y_lim = (-8, 8)
        shuffle = True
        X1_train = torch.tensor(
            make_moons(n_samples=n_samples_train, shuffle=shuffle, noise=noise)[0] * scale)
        X1_test = torch.tensor(
            make_moons(n_samples=n_samples_test, shuffle=shuffle, noise=noise)[0] * scale)
    elif dataset_name == "s-curve-sklearn":
        D = 2.0
        noise_ = 5e-3
        scale_ = 5.0
        x_lim = (-15, 15)
        y_lim = (-12, 12)
        X1_train = torch.tensor(make_s_curve(n_samples=n_samples_train, noise=noise_)[0][:, [0, 2]]) * scale_
        X1_test = torch.tensor(make_s_curve(n_samples=n_samples_test, noise=noise_)[0][:, [0, 2]]) * scale_
    elif dataset_name == "swissroll-3d-sklearn":
        D = 8.0
        noise_ = 0.005
        scale_ = 2.0
        x_lim = (-10, 10)
        y_lim = (-10, 10)
        X1_train = torch.tensor(make_swiss_roll(n_samples=n_samples_train, noise=noise_)[0] / scale_)
        X1_test = torch.tensor(make_swiss_roll(n_samples=n_samples_test, noise=noise_)[0] / scale_)
    elif dataset_name == "s-curve-3d-sklearn":
        D = 8.0
        noise_ = 0.005
        scale_ = 2.0
        x_lim = (-2.5, 5)
        y_lim = (-5, 5)
        # tns = torch.tensor([1., 2])
        # k = torch.get_default_dtype()
        X1_train = torch.tensor(make_s_curve(n_samples=n_samples_train, noise=noise_)[0])
        X1_test = torch.tensor(make_s_curve(n_samples=n_samples_test, noise=noise_)[0])
    else:
        raise NotImplementedError(f"Experiments not implemented for target '{dataset_name}'.")

    data_dict["D"] = D
    data_dict["x_lim"] = x_lim
    data_dict["y_lim"] = y_lim
    data_dict["X1_train"] = X1_train
    data_dict["X1_test"] = X1_test
    return data_dict
