import torch
from .sem_base import SEM


def linear_collider_lambda_us(us): return us
def linear_collider_lambda_x1(xs, u1): return xs + u1
def linear_collider_lambda_x2(xs, x1, u2): return 0.5 * xs + u2
def linear_collider_lambda_x3(xs, x1, x2, u3): return 0.05 * xs + 0.25 * x1 + 0.25 * x2 + u3
def linear_collider_lambda_y(xs, x1, x2, x3, u): return torch.special.expit(x1 + 0.5 * x2 + x3 + 0.5*u)


def linear_collider_inverse_xs(xs): return xs
def linear_collider_inverse_x1(xs, x1): return x1 - xs
def linear_collider_inverse_x2(xs, x1, x2): return x2 - 0.5 * xs
def linear_collider_inverse_x3(xs, x1, x2, x3): return x3 - 0.25 * x2 - 0.25 * x1 - 0.05 * xs


def nonlinear_collider_lambda_us(us): return us
def nonlinear_collider_lambda_x1(xs, u1): return 0.5 * xs + u1
def nonlinear_collider_lambda_x2(xs, x1, u2): return 0.5 * xs + u2
def nonlinear_collider_lambda_x3(xs, x1, x2, u3): return 0.25 * xs + 0.25 * x1 ** 2 + 0.25 * x2 ** 2 + u3
def nonlinear_collider_lambda_y(xs, x1, x2, x3, u): return torch.special.expit(x1**3 + x2 + 3*x3 + 0.5 * u)


def nonlinear_collider_inverse_xs(xs): return xs
def nonlinear_collider_inverse_x1(xs, x1): return x1 - 0.5 * xs
def nonlinear_collider_inverse_x2(xs, x1, x2): return x2 - 0.5 * xs
def nonlinear_collider_inverse_x3(xs, x1, x2, x3): return x3 - 0.25 * x2 ** 2 - 0.25 * x1 ** 2 - 0.25 * xs


class ColliderSens(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        label_eq = None
        if sem_name == "linear":
            functions = [
                linear_collider_lambda_us,
                linear_collider_lambda_x1,
                linear_collider_lambda_x2,
                linear_collider_lambda_x3
            ]
            inverses = [
                linear_collider_inverse_xs,
                linear_collider_inverse_x1,
                linear_collider_inverse_x2,
                linear_collider_inverse_x3
            ]
            label_eq = linear_collider_lambda_y
        elif sem_name == "non-linear":
            functions = [
                nonlinear_collider_lambda_us,
                nonlinear_collider_lambda_x1,
                nonlinear_collider_lambda_x2,
                nonlinear_collider_lambda_x3
            ]
            inverses = [
                nonlinear_collider_inverse_xs,
                nonlinear_collider_inverse_x1,
                nonlinear_collider_inverse_x2,
                nonlinear_collider_inverse_x3
            ]
            label_eq = nonlinear_collider_lambda_y
        super().__init__(functions, inverses, sem_name, label_eq)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((4, 4))

        adj[0, :] = torch.tensor([0, 0, 0, 0])
        adj[1, :] = torch.tensor([1, 0, 0, 0])
        adj[2, :] = torch.tensor([1, 0, 0, 0])
        adj[3, :] = torch.tensor([1, 1, 1, 0])
        if add_diag:
            adj += torch.eye(4)

        return adj

    def intervention_index_list(self):
        return [1]
