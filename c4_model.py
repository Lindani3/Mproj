"""
c4_model.py
===========
GRU surrogate model for Hull-White IRS valuation (all three variants).

Architecture
------------
The IRSSurrogate model processes the 20-length discount-factor sequence
through a multi-layer GRU, concatenates the final hidden state with three
scalar inputs (a, sigma, t_k), and maps to a scalar output via a linear layer:

    GRU(input_size=1, hidden_size=H, num_layers=L)
        input : (B, 20, 1)    P(0, T_j) at semi-annual T_PAY knots
        output: h_n[-1] of shape (B, H)

    fc : Linear(H + 3, 1)

All three models (1a, 1b, 2) share this architecture and use MSE loss.
Both analytical (1a) and Monte Carlo (1b) labels are trained with the same
objective so that differences in results are attributable solely to label
noise, not to any modelling choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IRSSurrogate(nn.Module):
    """
    GRU-based surrogate for the Hull-White payer IRS value.

    Parameters
    ----------
    n_yields   : length of discount-factor input sequence (20 semi-annual knots)
    n_scalar   : number of scalar covariates appended to GRU output (3: a, sigma, t_k)
    hidden_dim : GRU hidden state dimension H
    n_layers   : number of stacked GRU layers
    """

    def __init__(
        self,
        n_yields:   int = 20,
        n_scalar:   int = 3,
        hidden_dim: int = 128,
        n_layers:   int = 2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size  = 1,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = 0.1 if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + n_scalar, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_disc: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_disc   : (B, 20, 1)  discount-factor sequence
        x_scalar : (B, 3)      [a, sigma, t_k]

        Returns
        -------
        (B, 1) predicted value (E[V], E[V_MC], or EPE)
        """
        _, h_n   = self.gru(x_disc)          # h_n: (n_layers, B, H)
        h_last   = h_n[-1]                   # (B, H)
        combined = torch.cat([h_last, x_scalar], dim=-1)   # (B, H+3)
        return self.head(combined)           # (B, 1)


def mse_loss(pred: torch.Tensor, y_price: torch.Tensor) -> torch.Tensor:
    """MSE between predicted scalar values and price labels."""
    return F.mse_loss(pred.squeeze(-1), y_price)
