import torch

from .sem_base import SEM


# e_0, e_G, e_A = -1, 0.5, 1
# l_0, l_A, l_G = 1, 0.01, 1
# d_0, d_A, d_G, d_L = -1, 0.1, 2, 1
# i_0, i_A, i_G, i_GE = -4, 0.1, 2, 1
# s_0, s_I = -4, 1.5


def sgn_lambda(income, savings):
    return torch.where((income >= 0) & (savings >= 0), torch.tensor(1.0), torch.tensor(-1.0))


def loan_lambda_gender(u1): return u1
def loan_lambda_age(gender, u2): return -35 + u2
def loan_lambda_edu(gender, age, u3): return -0.5 + (1 + torch.exp(-(-1.0 + 0.5*gender + (1 + torch.exp(-0.1*age))**(-1) + u3)))**(-1)
def loan_lambda_loan_amount(gender, age, edu, u4): return 1 + 0.01*(age - 5)*(5 - age) + (1-gender) + u4
def loan_lambda_duration(gender, age, edu, loan_amount, u5): return -1 + 0.1*age + 2*(1-gender) + loan_amount + u5
# def loan_lambda_loan_amount_2(gender, age, edu, u4): return 1 + 0.01*(age - 5)*(5 - age) + (gender) + u4
# def loan_lambda_duration_2(gender, age, edu, loan_amount, u5): return -1 + 0.1*age + 2*(gender) + loan_amount + u5
def loan_lambda_income(gender, age, edu, loan_amount, duration, u6): return -4 + 0.1*(age + 35) + 2*gender + gender*edu + u6
def loan_lambda_savings(gender, age, edu, loan_amount, duration, income, u7): return -4 + 1.5*(income > 0)*income + u7
def loan_lambda_y(gender, age, edu, loan_amount, duration, income, savings, u): return torch.special.expit(0.3 * (-loan_amount -duration + income + savings + income*savings))
def loan_lambda_y_2(gender, age, edu, loan_amount, duration, income, savings, u): return torch.special.expit(0.3 * (-loan_amount -duration + income + savings + sgn_lambda(income, savings)*income*savings))


def loan_inverse_gender(gender): return gender
def loan_inverse_age(gender, age): return age + 35
def loan_inverse_edu(gender, age, edu): return torch.log((0.5+edu) / (0.5-edu)) + 1 - 0.5*gender - (1 + torch.exp(-0.1*age))**(-1)
def loan_inverse_loan_amount(gender, age, edu, loan_amount): return loan_amount - 1 - 0.01*(age-5)*(5-age) - (1-gender)
def loan_inverse_duration(gender, age, edu, loan_amount, duration): return duration + 1 - 0.1*age - 2*(1-gender) - loan_amount
# def loan_inverse_loan_amount_2(gender, age, edu, loan_amount): return loan_amount - 1 - 0.01*(age-5)*(5-age) - (gender)
# def loan_inverse_duration_2(gender, age, edu, loan_amount, duration): return duration + 1 - 0.1*age - 2*(gender) - loan_amount
def loan_inverse_income(gender, age, edu, loan_amount, duration, income): return income + 4 - 0.1*(age + 35) - 2*gender - gender*edu
def loan_inverse_savings(gender, age, edu, loan_amount, duration, income, savings): return savings + 4 - 1.5*(income > 0)*income


class Loan(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        label_eq = None
        if sem_name == "linear":
            functions = [
                loan_lambda_gender,
                loan_lambda_age,
                loan_lambda_edu,
                loan_lambda_loan_amount,
                loan_lambda_duration,
                loan_lambda_income,
                loan_lambda_savings
            ]
            inverses = [
                loan_inverse_gender,
                loan_inverse_age,
                loan_inverse_edu,
                loan_inverse_loan_amount,
                loan_inverse_duration,
                loan_inverse_income,
                loan_inverse_savings
            ]
            label_eq = loan_lambda_y
        elif sem_name == "non-linear":
            functions = [
                loan_lambda_gender,
                loan_lambda_age,
                loan_lambda_edu,
                loan_lambda_loan_amount,
                loan_lambda_duration,
                loan_lambda_income,
                loan_lambda_savings
            ]
            inverses = [
                loan_inverse_gender,
                loan_inverse_age,
                loan_inverse_edu,
                loan_inverse_loan_amount,
                loan_inverse_duration,
                loan_inverse_income,
                loan_inverse_savings
            ]
            label_eq = loan_lambda_y_2
        # elif sem_name == "non-linear":
        #     raise NotImplementedError

        super().__init__(functions, inverses, sem_name, label_eq)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((7, 7))

        adj[0, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        adj[1, :] = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        adj[2, :] = torch.tensor([1, 1, 0, 0, 0, 0, 0])
        adj[3, :] = torch.tensor([1, 1, 0, 0, 0, 0, 0])
        adj[4, :] = torch.tensor([1, 1, 0, 1, 0, 0, 0])
        adj[5, :] = torch.tensor([1, 1, 1, 0, 0, 0, 0])
        adj[6, :] = torch.tensor([0, 0, 0, 0, 0, 1, 0])
        if add_diag:
            adj += torch.eye(7)

        return adj

    def intervention_index_list(self):
        return [0, 1]
