"""
Script written by mbaddar to compare TT against NN-Recflow
---
Sample Run
python3 train_eval_linear_recflow.py --seed 42 --dataset-name swissroll-3d-sklearn --bases rbf --length-scale 0.5
    --n-centers 30 --odesolve-method-nn euler --odesolve-method-tt rk4 --rank 7 --verbose 2 ---
Sources
1. Weight Decay
https://discuss.pytorch.org/t/weight-decay-parameter/83023/2

2. Cuda/GPU slower than CPU
https://stackoverflow.com/a/52469317/5937273
https://www.mathworks.com/matlabcentral/answers/788519-why-is-my-code-running-slower-on-the-gpu

3. Changing type ( with x.type(...) )  with torch.cuda.FloatTensor or torch.cuda.DoubleTensor
https://discuss.pytorch.org/t/difference-between-setting-tensor-to-device-and-setting-dtype-to-cuda-floattensor/98658
quote:
"Changing the type to torch.cuda.FloatTensor would not only push the tensor to the default GPU but would also
potentially transform the data type.
The to('cuda:id') operation would only transform the tensor to the specified device as seen here:

x = torch.tensor([1])
print(x.type())
> torch.LongTensor

y = x.type(torch.cuda.FloatTensor)
print(y, y.type())
> tensor([1.], device='cuda:0') torch.cuda.FloatTensor

z = x.to('cuda')
print(z, z.type())
> tensor([1], device='cuda:0') torch.cuda.LongTensor"

---------
Notes
=========
Reproducibiliy has been tested
Run # 1
/home/mbaddar/Documents/mbaddar/phd/tt-flow-matching/.venv/bin/python /home/mbaddar/Documents/mbaddar/phd/tt-flow-matching/train_eval_linear_recflow.py --seed 42 --dataset-name circles-sklearn --bases rbf --length-scale 0.5 --n-centers 10 --odesolve-method-nn euler --odesolve-method-tt rk4 --rank 1 --verbose 2

2024-10-14 11:40:28.746 | INFO     | flow_matching.viz:plot_results:191 - Loss value for model : nn = 0.028401458850011878
2024-10-14 11:40:38.305 | INFO     | flow_matching.viz:plot_results:191 - Loss value for model : tt = 0.4331484736256501

2024-10-14 11:51:39.443 | INFO     | flow_matching.viz:plot_results:191 - Loss value for model : nn = 0.028401458850011878
2024-10-14 11:51:48.978 | INFO     | flow_matching.viz:plot_results:191 - Loss value for model : tt = 0.4331484736256501

Run # 2
/home/mbaddar/Documents/mbaddar/phd/tt-flow-matching/.venv/bin/python /home/mbaddar/Documents/mbaddar/phd/tt-flow-matching/train_eval_linear_recflow.py --seed 42 --dataset-name circles-sklearn --bases rbf --length-scale 0.5 --n-centers 10 --odesolve-method-nn euler --odesolve-method-tt rk4 --rank 2 --verbose 2

2024-10-14 12:10:50.367 | INFO     | flow_matching.viz:plot_results:191 - Loss value for model : nn = 0.028401458850011878
2024-10-14 12:10:59.460 | INFO     | flow_matching.viz:plot_results:191 - Loss value for model : tt = 0.37268298677733963


2024-10-14 12:16:59.783 | INFO     | flow_matching.viz:plot_results:191 - Loss value for model : nn = 0.028401458850011878
2024-10-14 12:17:09.109 | INFO     | flow_matching.viz:plot_results:191 - Loss value for model : tt = 0.37268298677733963
"""
import argparse
import os.path
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from geomloss import SamplesLoss
from loguru import logger
from matplotlib import animation
from flow_matching.dataset import generate_dataset, SUPPORTED_DATASETS, DATASETS_DIMENSIONS
from flow_matching.eval import ODESOLVE_METHODS
from flow_matching.hopt.hopt_helper import do_hopt
from flow_matching.inference import infer
from flow_matching.models import RectifiedFlowNN, MLP
from flow_matching.training import train_rectified_flow_nn, train_tt_recflow
from datetime import datetime
from flow_matching.viz import plot_results, plot_original_prior_posterior, plot_samples
from reproducibility import set_env
from utils import remove_outliers_range

BASES = ["bsplines", "rbf", "fourier", "legendre"]
BEST_MODEL_PATH = "models/best_models"
TT_RECFLOW_MODEL_NAME_SUFFIX = "tt_recflow_best_model.pkl"


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help="SEED", required=True)
    parser.add_argument('--dataset-name', type=str, help="Dataset name",
                        choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument('--odesolve-method-nn', type=str,
                        help="ODESOLVE method for nn model", choices=ODESOLVE_METHODS, required=True)
    parser.add_argument('--odesolve-method-tt', type=str, help="ODESOLVE method for tt model"
                        , required=True)
    parser.add_argument("--rank", type=int, default=3, required=False)
    parser.add_argument("--reg-coeff", type=float, default=1e-3, required=False)
    parser.add_argument('--bases', type=str, choices=BASES, help="basis function", required=True)
    parser.add_argument("--length-scale", type=float, required=False, default=1.0)  # For rbf basis
    parser.add_argument("--n-centers", type=int, required=False, default=20, help="Number of centers for rbf")
    # Why only add length scale as cmd args param? As rbf is the best basis so far!
    parser.add_argument("--remove-outliers", type=bool, default=False, required=False)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--tt-only", action="store_true")
    parser.add_argument("--hopt", action="store_true")
    parser.add_argument("--hopt-max-evals", type=int, default=100)

    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    parser.add_argument("--use-best-model", action="store_true")
    return parser


def check_args(parsed_args):
    assert parsed_args.hopt_max_evals >= 1, "hopt max-evals must be >=2"


# arg parsing
args = get_arg_parser().parse_args()
check_args(args)
# set dtype and device
logger.info(f"parsed args : {args}")

# Set device and dtype
"""
Some notes 
1. DoubleTensor = float64 and FloatTensor = float32
https://discuss.pytorch.org/t/floattensor-and-doubletensor/28553 

2. From experiments better to stick with DoubleTensor
3. dtype will hold information for tensor data type and device , see this piece of code
see https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_transport_blur.html 
code snippet dtype = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
this code wants more accuracy for sinkhorn calculation to be close to Wasserstein Distance, hence using 
DoubleTensor. 

"""
# TODO , cannot understand why use Double with GPU while it is know for being optimized for Float32
#   See https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_transport_blur.html
#    Code snippet use_cuda = torch.cuda.is_available()
#   #  N.B.: We use float64 numbers to get nice limits when blur -> +infinity
#   dtype = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

assert torch.cuda.is_available(), "We need CUDA/GPU ! "
logger.info(f"Is Cuda Available = {torch.cuda.is_available()}")

if args.device == "cuda" and args.dtype == "float32":
    dtype = torch.cuda.FloatTensor
elif args.device == "cuda" and args.dtype == "float64":
    dtype = torch.cuda.DoubleTensor
elif args.device == "cpu" and args.dtype == "float32":
    dtype = torch.FloatTensor
elif args.device == "cpu" and args.dtype == "float64":
    dtype = torch.DoubleTensor
else:
    raise ValueError(f"Unknown device and dtype combination : {args.device},{args.dtype}")

set_env(seed=args.seed)
# logger.info(f"Default device = {str(torch.get_default_device())}")
# logger.info(f"Default dtype = {str(torch.get_default_dtype())}")
# set reproducibility
##
try:
    import scienceplots

    plt.style.use("science")
except ModuleNotFoundError:
    logger.warning(
        "Package 'scienceplots' not found, using default matplotlib's style."
    )

if __name__ == "__main__":
    ################  Preliminaries  ##############################
    n_samples_train: int = 50_000
    n_samples_test = 2000
    N_time_steps = 2000
    # Generate model file name and path
    best_model_file_name = f"{args.dataset_name}_{TT_RECFLOW_MODEL_NAME_SUFFIX}"
    best_model_file_path = os.path.join(BEST_MODEL_PATH, best_model_file_name)
    # run timestamp
    run_timestamp = datetime.now().isoformat()

    # set datatime based on dataset
    if args.dataset_name in DATASETS_DIMENSIONS.keys():
        data_dim = DATASETS_DIMENSIONS.get(args.dataset_name)
        logger.info(f"Setting data_dim={data_dim} as args.dataset_name = {args.dataset_name}")
    else:
        raise ValueError(f"Dataset name = {args.dataset_name} is not listed in the DATASETS_DIMENSIONS dictionary")
    # nn params
    nn_hidden_dim = 100
    nn_recflow_time_steps = 100
    batch_size = 4096

    # Generate priors
    # we use a standard Gaussian as prior

    mu_prior = torch.zeros(data_dim)
    cov_prior = torch.eye(data_dim)
    samples_0_train = (torch.distributions.
                       MultivariateNormal(loc=mu_prior, covariance_matrix=cov_prior).
                       sample(sample_shape=torch.Size([n_samples_train]))).type(dtype)
    samples_0_test = (torch.distributions.
                      MultivariateNormal(loc=mu_prior, covariance_matrix=cov_prior).
                      sample(sample_shape=torch.Size([n_samples_test]))).type(dtype)

    # Set output paths
    OUTPUT_PATH = Path(__file__).parent.absolute() / "results" / args.dataset_name
    OUTPUT_PATH.mkdir(exist_ok=True)
    # load data
    data_dict = generate_dataset(dataset_name=args.dataset_name, n_samples_train=n_samples_train,
                                 n_samples_test=n_samples_test, X0_train=samples_0_train, X0_test=samples_0_test)
    samples_1_train = data_dict["X1_train"].type(dtype)
    samples_1_test = data_dict["X1_test"].type(dtype)

    #
    D = data_dict["D"]
    x_lim = data_dict["x_lim"]
    y_lim = data_dict["y_lim"]

    limits = (-D - 5, D + 5)
    logger.info(f"Target: {args.dataset_name}")
    # plot the samples
    logger.info(f"Plotting {n_samples_train} prior and posterior samples")
    if data_dim == 2:
        output_ext = "pdf"
    elif data_dim == 3:
        output_ext = "jpg"
    else:
        raise ValueError(f"Cannot set ext for plotting data with dim >2")
    plot_original_prior_posterior(X0=samples_0_train, X1=samples_1_train,
                                  output_path=os.path.join(OUTPUT_PATH, f"original.{output_ext}"), limits=limits)
    ################### Train baseline NN-Recflow ##########################################
    # if args.tt_only=True, then make a dry run for tt only, to make the script faster.
    # In this case, NN results are garbage don't consider them
    nn_train_iters = 1 if args.tt_only else 20_000
    x_pairs = torch.stack([samples_0_train, samples_1_train], dim=1)
    nn_regularization_coeff = 1e-6
    recflow_model_nn = RectifiedFlowNN(model=MLP(input_dim=data_dim, hidden_num=nn_hidden_dim).type(dtype),
                                       num_time_steps=nn_recflow_time_steps)
    # get memory size in terms of scalars
    recflow_model_nn_numel = recflow_model_nn.numel()
    optimizer = torch.optim.Adam(recflow_model_nn.model.parameters(), lr=5e-3, weight_decay=nn_regularization_coeff)
    start_time = datetime.now()
    _, loss_curve = train_rectified_flow_nn(rectified_flow_nn=recflow_model_nn, optimizer=optimizer, pairs=x_pairs,
                                            batchsize=4096, inner_iters=nn_train_iters)
    end_time = datetime.now()
    if args.tt_only:
        logger.info(f"args.tt_only = True, skipped NN-RecFlow training")
    else:
        logger.info(f"Training time for NN-Recflow with dtype {dtype} "
                    f"= {(end_time - start_time).seconds} seconds")
    start_time = datetime.now()
    traj = recflow_model_nn.sample_ode(z0=samples_0_test, N=N_time_steps, ode_solve_method=args.odesolve_method_nn)
    end_time = datetime.now()
    logger.info(f"Inference time for NN-RecFlow with dtype {dtype} "
                f"= {(end_time - start_time).seconds} seconds")
    z1_recflow_nn = traj[-1]
    logger.info(f"Training and inference for NN-RecFlow has finished successfully")
    # sys.exit(-1) # FIXME, just to exit when trying to test nn only
    ################### Samples and targets  #############################################
    logger.info("Computing the linear transformation of X1")
    # we are looking for the affine transformation Ax+b s.t.
    # - Cov(AX+b) is diagonal -> A=eigenvectors of cov(X) is a good choice
    # - E[AX+b]= 0 -> b=-AE[X] is a good choice
    sigma_X1 = torch.cov(samples_1_train.T)  # (d, d)
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma_X1)
    # A = 1.0 / torch.sqrt(eigenvalues) * eigenvectors.T
    # A = torch.eye(d)
    A = eigenvectors.T
    Ainv = torch.linalg.inv(A)
    # b = 0
    b = -A @ torch.mean(samples_1_train, dim=0)
    X0, X1_transformed = samples_0_train, (samples_1_train @ A.T + b)

    logger.info("Plotting the transformed samples")

    # plot the transformed samples
    plot_samples(X=X1_transformed, output_path=os.path.join(OUTPUT_PATH, f"transformed.{output_ext}"), limits=limits)
    ##############  Learn TT #################################
    """
    Params we got from Hyper-opt
    Hyperopt finished after 50 max-evaluations
    with min_loss = 0.035429421812295914 and best_params = {'l': np.float64(0.8786014577453837),
        'nc': np.int64(20), 'r': np.int64(4)} and took 49 seconds to finish
    
    """
    tt_rank = args.rank
    tt_als_iterations = 200
    tt_als_reg_coeff = args.reg_coeff
    tt_als_tol = 5e-10

    kwargs = dict()
    if args.bases == "bsplines":
        kwargs["degree"] = 2
        kwargs["n_knots"] = 30
    if args.bases == "rbf":
        length_scale = args.length_scale  # fixed length scale
        kwargs["length_scales"] = [length_scale] * (data_dim + 1)
        n_centres = args.n_centers
        kwargs["n_centres"] = n_centres

    else:
        raise ValueError(f"Unsupported basis model : {args.bases}")
    logger.info(f"Running with bases {args.bases} with kwargs : {kwargs}")

    # set training parameters: either fixed or from hopt run
    samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.99)
    if args.hopt:
        hopt_start_timestamp = datetime.now()
        logger.info(f"Run hyper-opt to get the best tt-recflow model")
        ftt_results_model, best_params, min_loss = do_hopt(X0_train=samples_0_train,
                                                           X0_test=samples_0_test, X1_train=samples_1_train,
                                                           X1_test=samples_1_test,
                                                           N_time_steps=N_time_steps,
                                                           basis_model=args.bases,
                                                           tol=tt_als_tol,
                                                           tt_als_iterations=tt_als_iterations, verbose=args.verbose,
                                                           ode_solve_method=args.odesolve_method_tt,
                                                           max_evals=args.hopt_max_evals,
                                                           samples_loss=samples_loss)

        hopt_end_time = datetime.now()
        tt_rank = best_params["r"]
        length_scale = best_params["l"]
        n_centres = best_params["nc"]
        tt_als_reg_coeff = best_params["reg_coeff"]
        logger.info(f"Hyperopt finished after {args.hopt_max_evals} max-evaluations "
                    f"with min_loss = {min_loss} and best_params = "
                    f"{best_params} and took {(hopt_end_time - hopt_start_timestamp).seconds} seconds to finish")
        logger.info(f"Saving the best model to {best_model_file_path}")
        with open(best_model_file_path, "wb") as f:
            pickle.dump(obj=ftt_results_model, file=f)
            f.flush()
            f.close()
        logger.info(f"Successfully saved best model to = {best_model_file_path}")
        # # FIXME, assuming rbf - make it more generic
        # # FIXME, undo the fixed param part
        # tt_rank = tt_rank  # best_params["r"]
        # kwargs["length_scales"] = [length_scale] * (data_dim + 1)  # [best_params["l"]] * (data_dim + 1)
        # kwargs["n_centers"] = args.n_centers  # best_params["nc"]
    elif not args.hopt and not args.use_best_model:
        logger.info(f"Getting TT-RecFlow model based on fixed params ")
        ftt_results_model = train_tt_recflow(rank=tt_rank, bases_model=args.bases, iterations=tt_als_iterations,
                                             reg_coeff=tt_als_reg_coeff, tol=tt_als_tol, X0=samples_0_train,
                                             X1=X1_transformed, verbose=args.verbose, N_time_steps=N_time_steps,
                                             **kwargs)
    elif args.use_best_model:
        with open(os.path.join(best_model_file_path), "rb") as f:
            ftt_results_model = pickle.load(f)
    else:
        raise ValueError(f"Cannot understand arg.param to generate/load models : args = {args}")
    # Get ftt model size.
    # Why after training ? as we might apply dynamic rank adaptation which can lead to
    # changing core sizes
    recflow_model_tt_numel = sum([ftt_model_element.size for ftt_model_element in ftt_results_model])
    # quick report for model numel (memory size)
    logger.info(f"***Memory Size Comparison***\n")
    logger.info(f"For NN-RecFlow , numel = {recflow_model_nn_numel}")
    logger.info(f"For TT-Recflow , numel = {recflow_model_tt_numel}")
    logger.info(
        f"Memory compression ratio = {float(recflow_model_tt_numel - recflow_model_nn_numel) / recflow_model_nn_numel}")
    # Recall that TT practical benefit it to "compress" the model memory with an acceptable tradeoff for accuracy
    # papers
    # 1. https://arxiv.org/abs/2101.11714
    # 2. https://arxiv.org/abs/2307.00526
    # 3. https://www.iccs-meeting.org/archive/iccs2022/papers/133520635.pdf

    ############## Inference ###################

    logger.info(f"TT-RecFlow : Inferring with N_time_steps = {N_time_steps} with method {args.odesolve_method_tt}")
    t_eval = torch.linspace(0.0, 1.0, steps=N_time_steps).type(dtype)
    start_time = datetime.now()

    traj = infer(
        ftt_results_model,
        # torch.from_numpy(rng.multivariate_normal(mu_prior, cov_prior, size=(2000,))),
        samples_0_test,
        t_eval=t_eval,
        # method="euler",
        method=args.odesolve_method_tt,
        # method="midpoint",
        # method="dopri5",
    )
    end_time = datetime.now()
    logger.info(f"Inference time for TT-RecFlow {(end_time - start_time).seconds} seconds")
    z1_recflow_tt = (traj[-1, :, :] - b) @ Ainv.T
    samples_1_train_min, samples_1_train_max = torch.min(input=samples_1_train, dim=0).values, torch.max(
        input=samples_1_train, dim=0).values
    z1_recflow_nn_filtered = remove_outliers_range(x=z1_recflow_nn, x_min=samples_1_train_min,
                                                   x_max=samples_1_train_max) if args.remove_outliers else z1_recflow_nn
    z1_recflow_tt_filtered = remove_outliers_range(x=z1_recflow_tt, x_min=samples_1_train_min,
                                                   x_max=samples_1_train_max) if args.remove_outliers else z1_recflow_tt
    if args.remove_outliers:
        logger.info(f"Shape of inferred z1 by NN-Recflow before and after outlier "
                    f"removal = {z1_recflow_nn.shape},{z1_recflow_nn_filtered.shape}")
        logger.info(f"Shape of inferred z1 by TT-Recflow before and after outlier "
                    f"removal = {z1_recflow_tt.shape},{z1_recflow_tt_filtered.shape}")
    else:
        logger.info(f"No outlier removal is applied")
    ######### Plot Generated Results and Metadata ###########
    generation_results_dict = {"nn": {"z1": z1_recflow_nn_filtered, "odesolve_method": args.odesolve_method_nn},
                               "tt": {"z1": z1_recflow_tt_filtered, "odesolve_method": args.odesolve_method_tt}}
    logger.info("Plotting Generation Results along with metadata")
    generated_suffix = f"recflow_nn_tt_rank_{tt_rank}"
    if args.bases == "rbf":  # why only rbf for suffix generation? because it is the best so far
        # scientific format for floats
        # https://www.scaler.com/topics/python-scientific-notation/
        # code snippet
        # scientific_notation = "{:e}".format(0.000009732)
        # print(scientific_notation)
        # output 9.732e-06
        generated_suffix += (
            f"_tt_als_rbf_l_{np.round(length_scale, 2)}_"
            f"m_{n_centres}_"
            f"reg_coeff_{"{:e}".format(tt_als_reg_coeff)}_"
            f"dtype_{str(z1_recflow_tt_filtered.dtype)}_"
            f"device_{str(z1_recflow_tt_filtered.device)}_"
            f"hopt_{args.hopt}_"
            f"max_evals_{args.hopt_max_evals}_"
            f"run_timestamp_{run_timestamp}")
    else:
        raise ValueError(f"bases : {args.bases} is not supported for suffix generation")
    logger.info(f"Plotting generated results")

    plot_results(generation_results=generation_results_dict, x1=samples_1_test,
                 output_path=os.path.join(OUTPUT_PATH, f"generated_{generated_suffix}.png"),
                 x_lim=x_lim, y_lim=y_lim,
                 samples_loss=samples_loss)

    logger.info("Plotting the generated samples")
    plt.figure(figsize=(4, 4))
    plt.xlim(*limits)
    plt.ylim(*limits)

    plt.scatter(traj[0, :, 0].detach().cpu().numpy(), traj[0, :, 1].detach().cpu().numpy(), label=r"$\pi_0$",
                alpha=0.15, rasterized=True)
    plt.scatter(
        samples_1_train[:, 0].detach().cpu().numpy(), samples_1_train[:, 1].detach().cpu().numpy(), label=r"$\pi_1$",
        alpha=0.15, rasterized=True
    )
    plt.scatter(
        z1_recflow_tt[:, 0].detach().cpu().numpy(), z1_recflow_tt[:, 1].detach().cpu().numpy(), label="Generated",
        alpha=0.15, rasterized=True
    )
    plt.legend()
    plt.title("Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "generated.pdf", dpi=300)
    # plt.show()

    logger.info("Plotting the trajectories")
    plt.figure(figsize=(4, 4))
    plt.axis("equal")
    for i in range(100):
        plt.scatter(
            traj[0, i, 0].detach().cpu().numpy(),
            traj[0, i, 1].detach().cpu().numpy(),
            color="orange",
            # alpha=0.15,
            rasterized=True,
        )
        plt.scatter(
            traj[-1, i, 0].cpu().numpy(),
            traj[-1, i, 1].cpu().numpy(),
            color="green",
            # alpha=0.15,
            rasterized=True,
        )
        plt.plot(traj[:, i, 0].detach().cpu().numpy(), traj[:, i, 1].detach().cpu().numpy(), color="blue", alpha=0.3)
    plt.title("Transport trajectories")
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH / "trajectories.pdf", dpi=300)
    # plt.show()

    # sample movement

    logger.info("Plotting the samples animation")

    fig, ax = plt.subplots()
    scat = ax.scatter(traj[0, :, 0].detach().cpu().numpy(), traj[0, :, 1].detach().cpu().numpy(), alpha=0.15)

    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_title(f"Samples (t={0.:.2f})")
    fig.tight_layout()


    def update_scatter(i: int):
        global scat
        # global cf
        _t = t_eval[i]
        scat.set_offsets(traj[i].detach().cpu().numpy())

        ax.set_title(f"Samples (t={_t:.2f})")
        return (scat,)


    anim = animation.FuncAnimation(
        fig, update_scatter, frames=torch.arange(0, len(t_eval), 5), blit=True
    )
    anim.save(OUTPUT_PATH / "animation_samples.gif")

    # velocity field
    if data_dim == 2:
        logger.info("Plotting the velocity field")
        grid = torch.linspace(*limits, steps=50)
        X, Y = torch.meshgrid(grid, grid, indexing="ij")
        inputs = torch.column_stack(
            [X.ravel(), Y.ravel(), 0 * torch.ones((X.shape[0] * X.shape[1],))]
        )
        inputs = inputs.type(dtype)
        Z = torch.column_stack([ftt(inputs) for ftt in ftt_results_model])  # .detach().cpu()
        Z /= torch.linalg.norm(Z, dim=1, keepdim=True)

        fig, ax = plt.subplots()
        Q = ax.quiver(
            X.detach().cpu().numpy(),
            Y.detach().cpu().numpy(),
            Z[:, 0].detach().cpu().numpy().reshape(X.shape),
            Z[:, 1].detach().cpu().numpy().reshape(Y.shape),
            scale_units="inches",
            scale=7,
        )
        # cf = ax.contourf(X, Y, torch.linalg.norm(Z, dim=1).reshape(X.shape))
        ax.grid()
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_title(f"Velocity field (t={0.:.2f})")
        fig.tight_layout()
        t = torch.linspace(0.0, 1.0, steps=100)


        def update_quiver(i: int):
            global Q
            # global cf
            _t = t[i]
            inputs = torch.column_stack(
                [X.ravel(), Y.ravel(), _t * torch.ones((X.shape[0] * X.shape[1],))]
            ).type(dtype)
            Z = torch.column_stack([ftt(inputs) for ftt in ftt_results_model])  # .detach().cpu()
            Z /= torch.linalg.norm(Z, dim=1, keepdim=True)
            Q.set_UVC(
                Z[:, 0].reshape(X.shape).detach().cpu().numpy(),
                Z[:, 1].reshape(Y.shape).detach().cpu().numpy(),
            )
            # for coll in cf.collections:
            #     coll.remove()
            # cf = ax.contourf(X, Y, torch.linalg.norm(Z, dim=1).reshape(X.shape))
            ax.set_title(f"Velocity field (t={_t:.2f})")
            return (Q,)
            # return (cf,)


        anim = animation.FuncAnimation(fig, update_quiver, frames=len(t), blit=True)
        anim.save(OUTPUT_PATH / "velocity_field.gif")
    else:
        logger.info(f"Cannot plot velocity for data_dim >2 , data_dim = {data_dim}")
