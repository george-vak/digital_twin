"""Persistence forecast: use x[t] to predict x[t+h], for any positive h."""

from torch import nn


class PersistenceModel(nn.Module):
    def forward(self, x):
        # Same batch × window × features input as GRUModel.
        return x[:, -1, :]
