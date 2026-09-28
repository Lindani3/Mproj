"""
c4_model_v2.py
==============
GRU surrogate — sequence-to-sequence design (Frode et al. 2022).

One forward pass maps (discount factor curve, a, sigma) to a full
21-element output profile covering all semi-annual monitoring dates.

Architecture
------------
    GRU(input_size=1, hidden_size=H, num_layers=L, batch_first=True)
        input : (B, 20, 1)   discount factor sequence
        output: h_n[-1]  of shape (B, H)

    head : Linear(H+2, 64) -> ReLU -> Linear(64, 21)
        input : cat(h_last, [a, sigma])  of shape (B, H+2)
        output: (B, 21)  predicted profile
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IRSSurrogateV2(nn.Module):
    """
    Sequence-to-sequence GRU surrogate for Hull-White IRS/EPE profiles.

    Parameters
    ----------
    n_yields   : length of discount-factor input sequence (20)
    n_scalar   : number of scalar covariates appended to GRU output (2: a, sigma)
    n_out      : number of output values (21 semi-annual monitoring dates)
    hidden_dim : GRU hidden state dimension H
    n_layers   : number of stacked GRU layers
    """

    def __init__(
        self,
        n_yields:   int = 20,
        n_scalar:   int = 2,
        n_out:      int = 21,
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
            nn.Linear(64, n_out),
        )

    def forward(self, x_disc: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_disc   : (B, 20, 1)  discount-factor sequence
        x_scalar : (B, 2)      [a, sigma]

        Returns
        -------
        (B, 21) predicted profile
        """
        _, h_n   = self.gru(x_disc)
        h_last   = h_n[-1]                                      # (B, H)
        combined = torch.cat([h_last, x_scalar], dim=-1)        # (B, H+2)
        return self.head(combined)                               # (B, 21)


def mse_loss(pred: torch.Tensor, y_profile: torch.Tensor) -> torch.Tensor:
    """MSE over all (batch, date) predictions."""
    return F.mse_loss(pred, y_profile)
