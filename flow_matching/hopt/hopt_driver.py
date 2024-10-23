"""
This script is to
1. Run Hyperopt to find the best parameters for TT-RecFlow Training
2. Train and Eval a TT-RecFLow Model on the found best parameters

Libraries for Hyeropt
https://hyperopt.github.io/hyperopt/
https://github.com/ARM-software/mango

"""
from argparse import ArgumentParser

import torch
from torch.distributions import MultivariateNormal

from flow_matching.dataset import generate_dataset
from flow_matching.hopt.hopt_helper import hopt_tt_recflow_obj_fn
from loguru import logger
from hyperopt import hp, fmin, tpe, Trials

from reproducibility import set_env

# set default data type
torch.set_default_dtype(torch.float64)
# parse args
parser = ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max-evals", type=int, required=True)
parser.add_argument("--verbose", type=int, required=True)
parser.add_argument("--dataset-name", type=str, required=True)
parser.add_argument("--dry", action="store_true")

args = parser.parse_args()
logger.info(f"Passed args = {args}")
rng = set_env(seed=args.seed, default_dtype=torch.float64)

if __name__ == "__main__":
    # Data
    # TODO first we focus in circles, then we test other dataset for hyperopt
    N_train = 50_000
    N_test = 2_000
    N_time_steps = 2000
    data_dim = 2

    mu_prior = torch.zeros(data_dim)
    cov_prior = torch.eye(data_dim)
    mvn_prior = MultivariateNormal(loc=mu_prior, covariance_matrix=cov_prior)
    X0_train = mvn_prior.sample(sample_shape=torch.Size([N_train]))
    X0_test = mvn_prior.sample(sample_shape=torch.Size([N_test]))
    data_dict = generate_dataset(dataset_name=args.dataset_name, n_samples_train=N_train, n_samples_test=N_test,
                                 X0_train=X0_train, X0_test=X0_test)
    X1_train = data_dict["X1_train"]
    X1_test = data_dict["X1_test"]

    # Mean and cov of x, to compare to a train_eval script for reproducibility
    x1_train_mean = torch.mean(input=X1_train, dim=0)
    x1_test_mean = torch.mean(input=X1_test, dim=0)

    x1_train_cov = torch.cov(input=X1_train.T)
    x1_test_cov = torch.cov(input=X1_test.T)

    # Hopt
    # fixed TT train args
    reg_coeff = 1e-3
    tt_rank = 3
    tt_train_iterations = 200
    tol = 5e-10
    rbf_length_scale = 0.5
    sinkhorn_scale = 0.9
    n_rbf_centres = 20
    if args.dry:
        logger.info(f"Dry Hopt run")
        space = {"l0": rbf_length_scale,
                 "l1": rbf_length_scale,
                 "l2": rbf_length_scale,
                 "r": tt_rank,
                 "reg_coeff": reg_coeff,
                 "n_centers": n_rbf_centres,
                 "X0_train": X0_train,
                 "X0_test": X0_test,
                 "X1_train": X1_train,
                 "X1_test": X1_test,
                 "N_time_steps": N_time_steps,
                 "bases_model": "rbf",
                 "sinkhorn_scale": sinkhorn_scale,
                 "tol": tol,
                 "iterations": tt_train_iterations,
                 "verbose": args.verbose}
    else:
        logger.info(f"Actual Hopt Run")
        space = {"l0": hp.uniform("l0", 1e-5, 2.0),
                 "l1": hp.uniform("l1", 1e-5, 2.0),
                 "l2": hp.uniform("l2", 1e-5, 2.0),
                 "r": tt_rank,
                 "reg_coeff": reg_coeff,
                 "n_centers": n_rbf_centres,
                 "X0_train": X0_train,
                 "X0_test": X0_test,
                 "X1_train": X1_train,
                 "X1_test": X1_test,
                 "N_time_steps": N_time_steps,
                 "bases_model": "rbf",
                 "sinkhorn_scale": sinkhorn_scale,
                 "tol": tol,
                 "iterations": tt_train_iterations,
                 "verbose": args.verbose}
    trials = Trials()
    best = fmin(fn=hopt_tt_recflow_obj_fn, space=space, algo=tpe.suggest, max_evals=args.max_evals, trials=trials)
    print(best)
"""
hyperopt results
1 ) Circles
100%|██████████| 50/50 [1:12:31<00:00, 87.03s/trial, best loss: 0.027870137666202454]
{'l0': np.float64(1.4374840745863628), 'l1': np.float64(0.6032717158240077), 'l2': np.float64(0.11846112269385657)}
"""
