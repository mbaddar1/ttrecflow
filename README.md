# Rectified Flows Generative Models using Low-Rank Tensor-Trains [In Progress Project]
This code is to experiment the realization of  [Genrative Rectified Flows Models](https://arxiv.org/abs/2209.03003) using [Low-Rank Tensor-Trains](https://epubs.siam.org/doi/10.1137/090752286)
__objective__ :
1. Reduce the Memory Complexity for the Model approximating the Velocity Function
2. Reduce the increased Runtime complexity 
3. Provide Results witn comparable quality to the Neural-Network based Velocity function

## Description

TBD


## Training and Hyper-parameters optimization
Now we can let the code optimize hyperparameters, get the best model 
according to some loss/selection criteria for the hyperopt code and then use it for
tt-recflow inference, compare it with a baseline nn-recflow

sample call
```
tt-flow-matching/train_eval_linear_recflow.py --seed 42 --dataset-name swissroll-3d-sklearn --bases rbf --odesolve-method-nn euler --odesolve-method-tt rk4 --verbose 2 --device cpu --dtype float64 --hopt --hopt-max-eval 20 
```
Results
* ![Generated Swissroll-3D image](https://github.com/mbaddar1/ttrecflow/blob/main/results/swissroll-3d-sklearn/generated_recflow_nn_tt_rank_4_tt_als_rbf_l_0.53_m_36_reg_coeff_6.921959e-02_dtype_torch.float64_device_cpu_hopt_True_max_evals_20_run_timestamp_2024-10-21T18%3A27%3A10.690891.png)

__Description__
* The left (blue) sub-figure is samples from the [SwissRoll-3D distribution](https://scikit-learn.org/1.5/modules/generated/sklearn.datasets.make_swiss_roll.html)
* The middle (red) sub-figure is sampels from the Neural-Network based Velocity Approximation Function
* The right (yellow) sub-figure is is samples from out Functional-Tensor Train approximation for the velocity function
* The [Sinkhorn-Distance](https://proceedings.neurips.cc/paper_files/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf) is divergence measure between to samples. The values show that our Tensor-Train based RecFlow give higher quality samples that Neural-Network based ones, with almost 60% Memory-Compression raito.

## Credits

- [Charles MIRANDA](https://github.com/chmda)
- [Janina SCHÜTTE](https://github.com/janinaschutte)
- [David SOMMER](https://github.com/dvdsmr)
- Martin EIGEL
- Mohamed Baddar : mbaddar2@gmail.com

## License

TODO
