import torch
from sklearn.datasets import make_circles

def remove_outliers_quantile(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Outlier removal based on 95% for each dimension

    x : torch.Tensor of the shape N X D
    """
    D = x.shape[1]
    N = x.shape[0]
    q_ = torch.quantile(input=x, q=torch.tensor([alpha, 1 - alpha], dtype=x.dtype), dim=1)
    filter_idx = torch.tensor(data=[True] * N)
    for d in range(D):
        q_min = q_[0, d].item()
        q_max = q_[1, d].item()
        idx1 = torch.greater_equal(input=x[:, d], other=q_min)
        idx2 = torch.less_equal(input=x[:, d], other=q_max)
        idx = torch.bitwise_and(idx1, idx2)
        filter_idx = torch.bitwise_and(idx, filter_idx)

    x_filter_idx = x[filter_idx, :]
    return x_filter_idx


def remove_outliers_range(x: torch.Tensor, x_min, x_max):
    """
        Outlier removal based on a given explicit range

        x : torch.Tensor of the shape N X D
        """
    D = x.shape[1]
    N = x.shape[0]
    filter_idx = torch.tensor(data=[True] * N)
    for d in range(D):
        idx1 = torch.greater_equal(input=x[:, d], other=x_min[d].item())
        idx2 = torch.less_equal(input=x[:, d], other=x_max[d].item())
        idx = torch.bitwise_and(idx1, idx2)
        filter_idx = torch.bitwise_and(idx, filter_idx)

    x_filter_idx = x[filter_idx, :]
    return x_filter_idx

if __name__ == '__main__':
    n_samples = 10000
    x_train = torch.tensor(make_circles(n_samples=n_samples, shuffle=True, noise=0.05, factor=0.3)[0])
    x_test = torch.tensor(make_circles(n_samples=n_samples, shuffle=True, noise=0.05, factor=0.3)[0])
    x_min, x_max = torch.min(input=x_train, dim=0).values, torch.max(input=x_train, dim=0).values
    # apply outlier removal
    x1 = remove_outliers_quantile(x_test, alpha=0.001)
    x2 = remove_outliers_range(x=x_test, x_min=x_min, x_max=x_max)
