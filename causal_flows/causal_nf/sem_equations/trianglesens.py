import torch
from .sem_base import SEM


def linear_triangle_lambda_us(us): return us
def linear_triangle_lambda_x1(xs, u1): return xs + u1
def linear_triangle_lambda_x2(xs, x1, u2): return x1 + u2
def linear_triangle_lambda_x3(xs, x1, x2, u3): return xs + 0.25 * x1 + 0.5 * x2 + u3
def linear_triangle_lambda_y(xs, x1, x2, x3, u): return torch.special.expit(x1 + 0.5 * x2 + x3 + 0.5 * u)


def linear_triangle_inverse_xs(xs): return xs
def linear_triangle_inverse_x1(xs, x1): return x1 - xs
def linear_triangle_inverse_x2(xs, x1, x2): return x2 - x1
def linear_triangle_inverse_x3(xs, x1, x2, x3): return x3 - 0.5 * x2 - 0.25 * x1 - xs


def nonlinear_triangle_lambda_us(us): return us
def nonlinear_triangle_lambda_x1(xs, u1): return xs + u1
def nonlinear_triangle_lambda_x2(xs, x1, u2): return -1 + 3 / (1 + torch.exp(-2 * x1)) + u2
def nonlinear_triangle_lambda_x3(xs, x1, x2, u3): return xs + 0.25 * x1 ** 2 + 0.5 * x2 + u3
def nonlinear_triangle_lambda_y(xs, x1, x2, x3, u): return torch.special.expit(-1 + 2**x1 + 0.75 * (x2**-3) + 0.1 * x3**2 + 0.5*u)


def nonlinear_triangle_inverse_xs(xs): return xs
def nonlinear_triangle_inverse_x1(xs, x1): return x1 - xs
def nonlinear_triangle_inverse_x2(xs, x1, x2): return x2 - 3 / (1 + torch.exp(-2 * x1)) + 1
def nonlinear_triangle_inverse_x3(xs, x1, x2, x3): return x3 - 0.5 * x2 - 0.25 * x1 ** 2 - xs


class TriangleSens(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        label_eq = None
        if sem_name == "linear":
            functions = [
                linear_triangle_lambda_us,
                linear_triangle_lambda_x1,
                linear_triangle_lambda_x2,
                linear_triangle_lambda_x3,

            ]
            inverses = [
                linear_triangle_inverse_xs,
                linear_triangle_inverse_x1,
                linear_triangle_inverse_x2,
                linear_triangle_inverse_x3
            ]
            label_eq = linear_triangle_lambda_y
        elif sem_name == "non-linear":
            functions = [
                nonlinear_triangle_lambda_us,
                nonlinear_triangle_lambda_x1,
                nonlinear_triangle_lambda_x2,
                nonlinear_triangle_lambda_x3
            ]
            inverses = [
                nonlinear_triangle_inverse_xs,
                nonlinear_triangle_inverse_x1,
                nonlinear_triangle_inverse_x2,
                nonlinear_triangle_inverse_x3
            ]
            label_eq = nonlinear_triangle_lambda_y

        super().__init__(functions, inverses, sem_name, label_eq)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((4, 4))

        adj[0, :] = torch.tensor([0, 0, 0, 0])
        adj[1, :] = torch.tensor([1, 0, 0, 0])
        adj[2, :] = torch.tensor([0, 1, 0, 0])
        adj[3, :] = torch.tensor([1, 1, 1, 0])
        if add_diag:
            adj += torch.eye(4)

        return adj

    def intervention_index_list(self):
        return [0, 1]
