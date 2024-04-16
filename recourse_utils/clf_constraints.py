from fairtorch import ConstraintLoss
import torch


class EqualOpportunityLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=(0, 1), alpha=1, p_norm=2):
        """loss of EOP fairness

        Args:
            sensitive_classes (tuple/list, optional): list of unique values of sensitive attribute. Defaults to (0, 1).
            alpha (int, optional): [description]. Defaults to 1.
            p_norm (int, optional): [description]. Defaults to 2.
        """
        self.sensitive_classes = list(sensitive_classes)
        self.n_class = len(sensitive_classes)
        super(EqualOpportunityLoss, self).__init__(
            n_class=self.n_class, alpha=alpha, p_norm=p_norm
        )
        self.n_constraints = 2 * self.n_class
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        for i in range(self.n_constraints):
            j = i % 2
            if j == 0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j - 1] = -1.0
                self.M[i, -1] = 1.0
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X, out, sensitive, y):
        expected_values_list = []
        for v in self.sensitive_classes:
            idx_true = (sensitive == v) * (y == 1)  # torch.bool
            expected_values_list.append(out[idx_true].mean())
        expected_values_list.append(out[y == 1].mean())
        return torch.stack(expected_values_list)

    def forward(self, X, out, sensitive, y=None):
        return super(EqualOpportunityLoss, self).forward(X, out, sensitive, y=y)
