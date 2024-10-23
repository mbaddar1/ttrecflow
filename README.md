# TT-Flow matching

Experiments with using [tensor-trains](https://epubs.siam.org/doi/10.1137/090752286) as ansatz for [flow matching](https://arxiv.org/abs/2210.02747).

## Description

Let $\pi_0$ and $\pi_1$ be two probability measures on $\mathbb R^d$. We aim to find a *coordinate transformation* $\phi$ such that $\pi_1 = \phi_\sharp \pi_0$.
Let $v_t$ be a vector field that can be used to construct a time-dependent diffeomorphic map, called a *flow* $\phi : [0,1] \times \mathbb R^d \to \mathbb R^d$, defined via the ODE:

```math
\begin{aligned}
    \frac{\partial \phi_t}{\partial t}(x) &= v_t(\phi_t(x)),\\
    \phi_0(x) &= x
\end{aligned}
```

We thus have a probability path $\pi_t := [\phi_t]_\sharp \pi_0$. One way to check if a vector field $v_t$ generates a probability path is using the *continuity equation*,

```math
\frac{\partial \pi_t}{\partial t}(x) = -\mathrm{div}(\pi_t(x)v_t(x))
```

The *flow matching objective* is thus defined as

```math
\mathbb E_{\substack{t \sim \mathcal U(0,1)\\ X_t \sim \pi_t}}[\|v_t(X_t) - u_\theta(t, X_t)\|_2^2]
```

and is minimized over the parameters $\theta$.

If we consider the linear interpolation between $X_0 \sim \pi_0$ and $X_1 \sim \pi_1$, we define,
```math
X_t := (1-t)X_0 + tX_1
```

And thus, $v_t = X_1-X_0$. The objective can be simplified to,

```math
\mathbb E_{\substack{t \sim \mathcal U(0,1)\\ X_0 \sim \pi_0\\ X_1 \sim \pi_1}}[\|X_1 - X_0 - u_\theta(t, (1-t)X_0 + tX_1)\|_2^2]
```

## Table of contents

- [Requirement](#requirement)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Credits](#credits)
- [License](#license)

## Requirement

- Python >= 3.9
- [torch](https://pytorch.org/)
- [scipy](https://scipy.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [loguru](https://github.com/Delgan/loguru)
- [pandas](https://pandas.pydata.org/)
- [jaxtyping](https://github.com/patrick-kidger/jaxtyping)
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
- [scienceplots](https://github.com/garrettj403/SciencePlots)

## Installation

### Clone

```sh
# HTTPS
git clone https://github.com/chmda/tt-flow-matching.git
# SSH
git clone git@github.com:chmda/tt-flow-matching.git
```

### [Optional] Create an environment

#### Using [uv](https://github.com/astral-sh/uv)

```sh
uv venv  # Create a virtual environment at .venv.

# On macOS and Linux.
source .venv/bin/activate

# On Windows.
.venv\Scripts\activate
```

#### Using venv
```sh
python -m venv .venv # Create a virtual environment at .venv.

# On macOS and Linux.
source .venv/bin/activate

# On Windows.
.venv\Scripts\activate
```

### Install packages

#### Using [uv](https://github.com/astral-sh/uv)

```sh
uv pip install -r requirements.txt  # Install from a requirements.txt file.
```

#### Using pip
```sh
pip install -r requirements.txt  # Install from a requirements.txt file.
```


## Usage

1. Get some data
```python
n_samples: int = ...
X_0: Float[Array, "n_samples d"] = ...
X_1: Float[Array, "n_samples d"] = ...
t: Float[Array, "n_samples"] = ...
```
2. Define the desired flow to match
```python
from flow_matching.flow import LinearInterpolation
flow = LinearInterpolation(T=1.0)
```
3. Make an initial guess and declare some learning parameters
```python
from flow_matching.tt import HermitePolynomials, FTT, TT
degrees = [5] * (d + 1)
dims = [degree + 1 for degree in degrees]
initial_ranks = [1] + [3] * (d) + [1]

domains = [limits for _ in range(d)] + [[0.0, 1.0]]
bases = [HermitePolynomials(degree) for degree in degrees]
initial_guess = [FTT(TT.rand(dims, initial_ranks), bases) for _ in range(d)]

# ALS parameters
reg_coeff = 1e-3
iterations = 100
tol = 5e-10

# rule = DoerflerAdaptivity(delta=1e-6, max_ranks=[5] * d, dims=dims, verbose=True)
rule = None
```
6. Learn the *vector field*
```python
from flow_matching import learn_flow_matching
result = learn_flow_matching(
    X0=X0,
    X1=X1,
    t=t,
    flow=flow,
    initial_guess=initial_guess,
    rank_adaptivity=rule,
    regularisation=reg_coeff,
    max_iters=iterations,
    tolerance=tol,
    verbose=2,
)
```
7. Do inference
```python
from flow_matching import infer
t_eval = torch.linspace(0.0, 1.0, steps=1000)
x: Float[Array, "n_test d"] = ...
traj = infer(
    result,
    x,
    t_eval=t_eval,
    # method="euler",
    method="rk4",
)
```

## Example



You can find an example script in [test_linear_interpolation.py](/test_linear_interpolation.py).


## Hyper-opt Implementation and calling
Now we can let the code optimize hyperparameters, get the best model 
according to some loss/selection criteria for the hyperopt code and then use it for
tt-recflow inference, compare it with a baseline nn-recflow

sample call
```
tt-flow-matching/train_eval_linear_recflow.py --seed 42 --dataset-name swissroll-3d-sklearn --bases rbf --odesolve-method-nn euler --odesolve-method-tt rk4 --verbose 2 --device cpu --dtype float64 --hopt --hopt-max-eval 20 
```
Results
* Generated Figure :![generated_recflow_nn_tt_rank_4_tt_als_rbf_l_0.53_m_36_reg_coeff_6.921959e-02_dtype_torch.float64_device_cpu_hopt_True_max_evals_20_run_timestamp_2024-10-21T18:27:10.690891.png](results/swissroll-3d-sklearn/generated_recflow_nn_tt_rank_4_tt_als_rbf_l_0.53_m_36_reg_coeff_6.921959e-02_dtype_torch.float64_device_cpu_hopt_True_max_evals_20_run_timestamp_2024-10-21T18%3A27%3A10.690891.png)
* Best Model : [swissroll-3d-sklearn_tt_recflow_best_model.pkl](models/best_models/swissroll-3d-sklearn_tt_recflow_best_model.pkl)

## Credits

- [Charles MIRANDA](https://github.com/chmda)
- [Janina SCHÜTTE](https://github.com/janinaschutte)
- [David SOMMER](https://github.com/dvdsmr)
- Mohamed Baddar : mbaddar2@gmail.com
- Martin EIGEL

## License

TODO
