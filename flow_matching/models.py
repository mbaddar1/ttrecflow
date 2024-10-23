import torch
import numpy as np
from torchdiffeq import odeint


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_num: int):
        super().__init__()
        # self.bn = torch.nn.BatchNorm1d(num_features=input_dim+1)
        # Tried BatchNorm , did not lead to good generation results
        self.fc1 = torch.nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = torch.nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = torch.nn.Linear(hidden_num, input_dim, bias=True)
        self.act = torch.nn.ReLU()  # lambda x: torch.nn.ReLU()(x)
        self.model = torch.nn.Sequential(self.fc1, self.act, self.fc2, self.act, self.fc3)
        self.model = self.model

    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        x = inputs
        y = self.model(x)
        return y
        # x = self.bn(x)
        # x = self.fc1(x)
        # x = self.act(x)
        # x = self.fc2(x)
        # x = self.act(x)
        # x = self.fc3(x)

    def norm(self):
        norm_val = torch.sum(torch.tensor([torch.norm(param) for param in self.parameters()]))
        return norm_val

    def numel(self):
        tot_numel = 0
        for name, param in self.model.named_parameters():
            tot_numel += torch.numel(param)
        return tot_numel


class RectifiedFlowNN:
    def __init__(self, model=None, num_time_steps=1000):
        self.model = model
        self.N = num_time_steps

    # FIXME , repeated fn , need to make a base class for Recflow
    # @torch.no_grad()
    # def sample_ode(self, z0=None, N=None):
    #     # NOTE: Use Euler method to sample from the learned flow
    #     if N is None:
    #         N = self.N
    #     dt = 1. / N
    #     traj = []  # to store the trajectory
    #     z = z0.detach().clone()
    #     batchsize = z.shape[0]
    #
    #     traj.append(z.detach().clone())
    #     for i in tqdm(range(N), desc="Generating Trajectory"):
    #         t = torch.ones((batchsize, 1)) * i / N
    #         pred = self.model(z, t)
    #         z = z.detach().clone() + pred * dt
    #         traj.append(z.detach().clone())
    #     return traj
    @torch.no_grad()
    def sample_ode(self, z0, N, ode_solve_method):
        def f(t, z):
            B = z.shape[0]
            t_ = torch.tensor([t] * B).view(-1, 1).type(z.dtype).to(z.device)
            return self.model(z, t_)

        t_evals = torch.tensor(np.linspace(start=0.0, stop=1.0, num=N, endpoint=True)).type(z0.dtype).to(z0.device)
        traj = odeint(func=f, y0=z0, t=t_evals, method=ode_solve_method)
        return traj

    def numel(self):
        return self.model.numel()
