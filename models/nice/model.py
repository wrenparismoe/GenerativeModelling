import torch
import torch.nn as nn
from .layers import CouplingLayer


class NICE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(NICE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Sequence of 4 alternativing parity coupling layers
        self.couple1 = CouplingLayer(input_dim, hidden_dim, num_layers, parity="odd")
        self.couple2 = CouplingLayer(input_dim, hidden_dim, num_layers, parity="even")
        self.couple3 = CouplingLayer(input_dim, hidden_dim, num_layers, parity="odd")
        self.couple4 = CouplingLayer(input_dim, hidden_dim, num_layers, parity="even")

        """
        Create the scaling diagonal matrix (see Section 3.3 of the NICE paper)
        Multiplies the ith output value by S_ii. Weights certain dim more than others.
        Similar to eigenspectrum of PCA, exposing the variation present in each latent dimension 
        (larger S_ii means the less important dimension i is). More important dimensions of the 
        spctrum can be viewed as a manifold learned by the model.
        """
        self.s = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        """
        Forward pass is the encoding step of the NICE model.
        """
        # Apply the coupling layers
        y = self.couple1(x)
        y = self.couple2(y)
        y = self.couple3(y)
        y = self.couple4(y)
        # Apply the scaling layer
        # y = torch.matmul(y, torch.diag(torch.exp(self.s)))
        y = y * torch.exp(self.s)
        log_jacobian = torch.sum(self.s)

        return y, log_jacobian

    def inverse(self, y):
        """
        Inverse pass is the decoding step of the NICE model.
        """
        with torch.no_grad():
            # Apply the inverse scaling layer
            # x = torch.matmul(y, torch.diag(torch.exp(-self.s)))
            x = y / torch.exp(self.s)
            # Apply the inverse coupling layers
            x = self.couple4.inverse(x)
            x = self.couple3.inverse(x)
            x = self.couple2.inverse(x)
            x = self.couple1.inverse(x)
        return x
