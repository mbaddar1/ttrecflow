from copyreg import pickle
import pickle

import numpy as np
import ot
from ot.backend import torch

from flow_matching.viz import plot_results
from scipy.stats import wasserstein_distance_nd
import torch

if __name__ == "__main__":
    file = "/home/mbaddar/Documents/mbaddar/phd/tt-flow-matching/plot_params_circles-sklearn.pkl"
    with open(file, "rb") as f:
        params = pickle.load(f)
    # print(params)
    plot_results(generation_results=params["generation_results_dict"], x1=params["x1"],
                 output_path="mock_plot.png",
                 x_lim=params["x_lim"], y_lim=params["y_lim"],
                 samples_loss=params["loss"])
    x_ref = params["x1"].detach().cpu().numpy()
    x_tt = params["generation_results_dict"]["tt"]["z1"].detach().cpu().numpy()
    x_nn = params["generation_results_dict"]["nn"]["z1"].detach().cpu().numpy()
    # l1 = params["loss"](params["x1"].to(torch.float64),
    #                     params["generation_results_dict"]["tt"]["z1"].to(torch.float64))
    # print(l1)

    n = x_ref.shape[0]  # nb samples

    # mu_s = np.array([0, 0])
    # cov_s = np.array([[1, 0], [0, 1]])
    #
    # mu_t = np.array([4, 4])
    # cov_t = np.array([[1, -.8], [-.8, 1]])
    #
    # xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
    # xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

    # a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

    # loss matrix
    M = ot.dist(x_ref, x_ref)
    a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
    Gs = ot.sinkhorn2(a, b, M, 0.1)
    print(Gs)
    """
    OT
    nn : 0.11493280335103709
    tt : 0.11080725282607072
    ref : 0.0513251035141643 
    """
