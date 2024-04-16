import torch
from .sem_base import SEM


def linear_chain_lambda_us(us): return us
def linear_chain_lambda_x1(xs, u1): return xs + u1
def linear_chain_lambda_x2(xs, x1, u2): return xs + 0.5 * x1 + u2
def linear_chain_lambda_x3(xs, x1, x2, u3): return 0.25 * xs + 0.5 * x2 + u3
def linear_chain_lambda_y(xs, x1, x2, x3, u): return torch.special.expit(x1 + 0.5 * x2 + x3 + 0.5*u)


def linear_chain_inverse_xs(xs): return xs
def linear_chain_inverse_x1(xs, x1): return x1 - xs
def linear_chain_inverse_x2(xs, x1, x2): return x2 - 0.5 * x1 - xs
def linear_chain_inverse_x3(xs, x1, x2, x3): return x3 - 0.5 * x2 - 0.25 * xs


def nonlinear_chain_lambda_us(us): return us
def nonlinear_chain_lambda_x1(xs, u1): return xs + u1
def nonlinear_chain_lambda_x2(xs, x1, u2): return xs + 3 / (1 + torch.exp(-2 * x1)) + u2
def nonlinear_chain_lambda_x3(xs, x1, x2, u3): return 0.25 * xs + x2**2 + u3
def nonlinear_chain_lambda_y(xs, x1, x2, x3, u): return torch.special.expit(x1**3 + x2 + 0.75 * torch.exp(x3) + 0.5 * u - 50)


def nonlinear_chain_inverse_xs(xs): return xs
def nonlinear_chain_inverse_x1(xs, x1): return x1 - xs
def nonlinear_chain_inverse_x2(xs, x1, x2): return x2 - 3 / (1 + torch.exp(-2 * x1)) - xs
def nonlinear_chain_inverse_x3(xs, x1, x2, x3): return x3 - x2 ** 2 - 0.25 * xs


class ChainSens(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        label_eq = None
        if sem_name == "linear":
            functions = [
                linear_chain_lambda_us,
                linear_chain_lambda_x1,
                linear_chain_lambda_x2,
                linear_chain_lambda_x3
            ]
            inverses = [
                linear_chain_inverse_xs,
                linear_chain_inverse_x1,
                linear_chain_inverse_x2,
                linear_chain_inverse_x3
            ]
            label_eq = linear_chain_lambda_y
        elif sem_name == "non-linear":
            functions = [
                nonlinear_chain_lambda_us,
                nonlinear_chain_lambda_x1,
                nonlinear_chain_lambda_x2,
                nonlinear_chain_lambda_x3
            ]
            inverses = [
                nonlinear_chain_inverse_xs,
                nonlinear_chain_inverse_x1,
                nonlinear_chain_inverse_x2,
                nonlinear_chain_inverse_x3
            ]
            label_eq = nonlinear_chain_lambda_y

        super().__init__(functions, inverses, sem_name, label_eq)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((4, 4))

        adj[0, :] = torch.tensor([0, 0, 0, 0])
        adj[1, :] = torch.tensor([1, 0, 0, 0])
        adj[2, :] = torch.tensor([1, 1, 0, 0])
        adj[3, :] = torch.tensor([1, 0, 1, 0])
        if add_diag:
            adj += torch.eye(4)

        return adj

    def intervention_index_list(self):
        return [0, 1]
