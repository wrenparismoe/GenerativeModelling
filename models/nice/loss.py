import numpy as np
import torch
import torch.nn as nn

from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transforms import AffineTransform

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


class StandardLogisticDistribution:
    def __init__(self, data_dim=28 * 28, device="cpu"):
        self.m = TransformedDistribution(
            Uniform(
                torch.zeros(data_dim, device=device),
                torch.ones(data_dim, device=device),
            ),
            [
                SigmoidTransform().inv,
                AffineTransform(
                    torch.zeros(data_dim, device=device),
                    torch.ones(data_dim, device=device),
                ),
            ],
        )

    def log_pdf(self, z):
        return self.m.log_prob(z).sum(dim=1)

    def sample(self):
        return self.m.sample()


class _NICECriterion(nn.Module):
    """
    Implementation of equation (3) above. Base class for Gaussian and Logistic criterion classes.
    """

    def __init__(self, average=False, eps=1e-7):
        super(_NICECriterion, self).__init__()
        self.average = average
        self.eps = eps

    def prior(self, h):
        # Implement in child classes (4) and (5)
        raise NotImplementedError("Must implement prior function in child class")

    def forward(self, h, s):
        # Implementation of (3). Identical for both Gaussian and Logistic.
        log_likelihood = self.prior(h) + s
        if self.average:
            return -log_likelihood.mean()
        else:
            return -log_likelihood.sum()


class GaussianNICECriterion(_NICECriterion):
    """
    Implementation of (4) above. Gaussian prior based log-likelihood critereon.
    """

    def prior(self, h):
        return -0.5 * (
            torch.sum(torch.pow(h, 2), dim=1) + torch.log(torch.tensor(2 * np.pi))
        )


class LogisticNICECriterion(_NICECriterion):
    """
    Implementation of (5) above. Logistic prior based log-likelihood critereon.
    """

    def prior(self, h):
        # Implementation of (5) above.
        return -torch.sum(torch.log1p(torch.exp(h)) + torch.log1p(torch.exp(-h)), dim=1)
