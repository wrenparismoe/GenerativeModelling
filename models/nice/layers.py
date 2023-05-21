import torch
import torch.nn as nn


class ReLUNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim=1000, num_layers=5):
        super(ReLUNet, self).__init__()
        # latend_dim = input_dim // 2 (split into either odds or evens)
        # m: R^d -> R^{D-d}=R^{d} (ie: latent_dim -> input_dim - latent_dim = latent_dim)
        modules = [nn.Linear(latent_dim, hidden_dim)]
        for _ in range(num_layers):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class CouplingLayer(nn.Module):
    """
    Additive coupling layer for nice model:
        g(a;b) = (a + b), where
        a = x2, b = m(x1), and m is a recfified neural network (ReLUNet) from R^d -> R^{D-d}=R^{d}
        Note: x1 and x2 are the odd and even parts of the input x, so for x in R^{D}, x1, x2 in R^{d}
            d = D-d = D/2
        Forward:
            y1 = x1
            y2 = g(x2;m(x1)) = x2 + m(x1)
            y = (y1, y2)
        Inverse:
            x1 = y1
            x2 = g^{-1}(y2;m(y1)) = y2 - m(y1)
            x = (x1, x2)

    """

    def __init__(self, input_dim, hidden_dim, num_layers, parity: str):
        super(CouplingLayer, self).__init__()

        self.parity = parity
        latent_dim = input_dim // 2

        # Define NN layers for the transformation
        self.m = ReLUNet(latent_dim, hidden_dim, num_layers)

    def forward(self, x):
        # Split input into "odd" and "even" parts
        odd, even = x[:, 0::2], x[:, 1::2]
        if self.parity == "odd":
            x1, x2 = odd, even
        else:
            x1, x2 = even, odd

        # Part 1 of the input is pass through an identity function (remains unchanged)
        y1 = x1
        # Apply the coupling transformation to the part 2 of the input
        y2 = x2 + self.m(x1)

        # Concatenate (or couple) the two parts back together
        y = torch.cat([y1, y2], dim=1)
        return y

    def inverse(self, y):
        # Split the output into two parts
        odd, even = y[:, 0::2], y[:, 1::2]
        if self.parity == "odd":
            y1, y2 = odd, even
        else:
            y1, y2 = even, odd

        # The first part of the output is unchanged
        x1 = y1
        # Apply the inverse transformation to the second part of the output
        x2 = y2 - self.m(y1)

        # Concatenate the two parts back together
        x = torch.cat([x1, x2], dim=1)
        return x
