import numpy as np
import torch


class TorchStepFunction:
    """
    A basic step function.

    Values at the ends are handled in the simplest way possible:
    everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Parameters
    ----------
    x : array_like
    y : array_like
    ival : float
        ival is the value given to the values to the left of x[0]. Default
        is 0.
    is_sorted : bool
        Default is False.
    side : {'left', 'right'}, optional
        Default is 'left'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].
    """
    def __init__(self, x, y, ival=torch.tensor(0.), is_sorted=False, side='left'):
        if side.lower() not in ['right', 'left']:
            raise ValueError("side can only be left or right in StepFunction!")
        self.side = side
        _x, _y = x, y
        if not isinstance(x, torch.Tensor):
            _x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            _y = torch.tensor(y)
        if _x.size() != _y.size():
            raise ValueError("x and y must be same shape in StepFunction!")
        if len(_x.size()) != 1:
            raise ValueError("x and y must be 1-dim in StepFunction!")
        self.x = torch.cat((torch.tensor([float('inf')]), _x))
        self.y = torch.cat((torch.tensor([ival]), _y))

        if not is_sorted:
            self.x, _ = torch.sort(self.x)
            self.y, _ = torch.sort(self.y)
        self.x.requires_grad = True
        self.y.requires_grad = True
        self.n = self.x.size()[0]

    def __call__(self, time):
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time)
        tind = torch.searchsorted(self.x, time, right=(self.side == 'right')) - 1
        # TODO: Check if indexing works
        return self.y[tind]


class TorchECDF(TorchStepFunction):
    """
    Return the Empirical CDF of an array as a step function.

    Parameters
    ----------
    x : array_like
        Observations
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Empirical CDF as a step function.
    """
    def __init__(self, x, side='right'):
        # assert isinstance(x, torch.Tensor), "x must be tensor in TorchECDF!"
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = torch.clone(x)
        x, _ = torch.sort(x)
        nobs = len(x)
        y = torch.linspace(1. / nobs, 1, nobs)
        super(TorchECDF, self).__init__(x, y, side=side, is_sorted=True)


def feat_stats(preparator):
    if preparator.scaler is None:
        scaler = preparator.get_scaler(fit=True)
    else:
        scaler = preparator.scaler
    train_dataset = preparator.datasets[0]
    train_data = torch.tensor(train_dataset.X).type(torch.float)
    data_train = scaler.transform(train_data)
    cost_util_array = []
    num_features = data_train.shape[1]
    for idx in range(num_features):
        data_ = data_train[:, idx]
        ecdf = TorchECDF(data_)
        cost_util_array.append(('delta', ecdf, torch.max(data_), torch.min(data_), torch.mean(data_), torch.std(data_)))
    return cost_util_array


def reward_util_helper(data_array, data_likelihoods):
    # This array will be a list of ECDF objects and/or cost matrices.
    cost_util_array = []
    # For now we just do ECDF for all features. We later loop through likelihoods and append.
    num_features = data_array.shape[1]
    for idx in range(num_features):
        likelihood_ = data_likelihoods[idx][0]
        likelihood_name_, domain_size = likelihood_.name, likelihood_.domain_size
        if likelihood_name_ == 'delta':
            data_ = data_array[:, idx]
            ecdf = TorchECDF(data_)
            cost_util_array.append(('delta', ecdf, max(data_), min(data_), np.mean(data_), np.std(data_)))
        elif likelihood_name_ == 'ber':
            cost_matrix = torch.ones(2, 2)
            cost_matrix.fill_diagonal_(0)  # Diagonals should be 0, as no cost with no change
            cost_util_array.append(('ber', cost_matrix, torch.max(cost_matrix).item(), torch.min(cost_matrix).item(),
                                    torch.mean(cost_matrix).item(), torch.std(cost_matrix).item()))
        elif likelihood_name_ == 'cat':
            cost_matrix = torch.ones(domain_size, domain_size)
            cost_matrix.fill_diagonal_(0)
            cost_util_array.append(('cat', cost_matrix, torch.max(cost_matrix),
                                    torch.min(cost_matrix), np.mean(cost_matrix), np.std(cost_matrix)))
    return cost_util_array
