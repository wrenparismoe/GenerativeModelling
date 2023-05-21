import torch
import torch.nn as nn
import numpy as np

"""
Goal: LEarn a continuous, differentiable non-linear transformation f of the data distribution into
a simpler distribution via maximum log-likelihood given by the change of varaibles formula:
    log(p(x)) = log(p_H(f(x))) + log(|det(J_f(x))|)     (1)
where p_H(x), the prior distribution, is a predefined density function (Gaussian, logistic, etc.).
Then, building from (1), NICE criterion has following form:
    log(p(x)) = sum_d^D log(p_H_d(f_d(x))) + log(|det(J_f(x))|)     (2)
where f(x) = (f_d(x))_{d<=D} is the dth component of the transformation f, and p_H_d is the dth 
component of the prior p_H. The additive coupling layers have unit Jacobian determinant, and so do 
their compositions. So, the diagonal scaling matrix S is included such that 
S_ii: (x_i)_{i<=D} -> (S_ii x_i)_{i<=D}. Then, the NICE criterion becomes:
    log(p(x)) = sum_i^D [log(p_H_i(f_i(x))) + log(|S_ii|)     (3)
Here, the prior encourages S_ii to be small, while determinant log(|S_ii|) prevents S_ii from ever 
reaching zero. The prior distributuion is chosen to be factorial, i.e. p_H(y) = prod_d^D p_H_d(h_d),
and we define two loss functions for the NICE model, one for the Gaussian prior:
    log(p_H_d) = -1/2 (h_d^2) + log(2pi))     (4)
and one for the logistic prior:
    log(p_H_d) = -log(1 + exp(h_d)) - log(1 + exp(-h_d))     (5)
Note: NICE paper suggests using logistic prior as it provides a better behaving gradient.
"""


class _NICECriterion(nn.Module):
    """
    Implementation of equation (3) above. Base class for Gaussian and Logistic criterion classes.
    """

    def __init__(self, average=True):
        super(_NICECriterion, self).__init__()
        self.average = average

    def prior(self, h):
        # Implement in child classes (4) and (5)
        raise NotImplementedError("Must implement prior function in child class")

    def forward(self, h, s_diag):
        # Implementation of (3). Identical for both Gaussian and Logistic.
        # Don't take log of S_ii since it's already in log space, we take exp(S_ii) in forward pass.
        log_p = torch.sum(self.prior(h), dim=1) + torch.sum(s_diag)
        if self.average:
            return torch.mean(log_p)
        else:
            return torch.sum(log_p)


class GaussianNICECriterion(_NICECriterion):
    """
    Implementation of (4) above. Gaussian prior based log-lokielihood critereon.
    """

    def __init__(self, average=True):
        super(GaussianNICECriterion, self).__init__()

    def prior(self, h):
        # Implementation of (4) above.
        return -0.5 * (h**2 + torch.log(torch.tensor(2 * np.pi)))
