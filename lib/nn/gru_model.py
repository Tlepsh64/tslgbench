"""
GRU Model for spatiotemporal forecasting without graph structure.

A simple baseline that processes temporal patterns with GRU encoder
and applies a linear readout layer, ignoring spatial graph structure.
"""

import torch
import torch.nn as nn
from tsl.nn.models import BaseModel
from tsl.nn.blocks.encoders import RNN


class GRUModel(BaseModel):
    """
    Pure GRU model without graph convolutions.

    Processes temporal information with GRU encoder and applies
    a linear readout layer to produce forecasts. Ignores graph
    connectivity arguments passed by the predictor.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 output_size: int = 1,
                 horizon: int = 1,
                 n_nodes: int = None,
                 enc_layers: int = 2,
                 dropout: float = 0.0,
                 **kwargs):
        """
        Args:
            input_size: Number of input features per node per timestep
            hidden_size: Size of hidden representations (default: 64)
            output_size: Number of output features per node (default: 1)
            horizon: Number of steps to forecast (default: 1)
            n_nodes: Number of nodes in the graph (optional, unused)
            enc_layers: Number of GRU layers in encoder (default: 2)
            dropout: Dropout rate (default: 0.0)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.horizon = horizon
        self.n_nodes = n_nodes

        # GRU encoder: processes temporal dimension
        # Input: (batch, time, nodes, features)
        # Output: (batch, nodes, hidden_size) when return_only_last_state=True
        self.encoder = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=enc_layers,
            return_only_last_state=True,
            cell='gru',
            dropout=dropout
        )

        # Linear readout: projects hidden state to predictions
        # Input: (batch, nodes, hidden_size)
        # Output: (batch, nodes, horizon * output_size)
        self.readout = nn.Linear(hidden_size, output_size * horizon)

    def forward(self, x, edge_index=None, edge_weight=None, disjoint=False, **kwargs):
        """
        Forward pass of GRU model.

        Args:
            x: Input tensor of shape (batch, time, nodes, features)
            edge_index: Graph connectivity (ignored, for compatibility)
            edge_weight: Edge weights (ignored, for compatibility)
            disjoint: Disjoint batching flag (ignored, for compatibility)
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            Predictions of shape (batch, horizon, nodes, output_size)
        """
        # x shape: (batch, time, nodes, features)
        batch_size = x.size(0)
        n_nodes = x.size(2)

        # Encode temporal information with GRU
        # RNN expects (batch, time, nodes, features)
        # Returns (batch, nodes, hidden_size) with return_only_last_state=True
        h = self.encoder(x)

        # Apply linear readout to each node independently
        # h shape: (batch, nodes, hidden_size)
        # out shape: (batch, nodes, horizon * output_size)
        out = self.readout(h)

        # Reshape output to (batch, horizon, nodes, output_size)
        out = out.view(batch_size, n_nodes, self.horizon, self.output_size)
        out = out.permute(0, 2, 1, 3)  # (batch, horizon, nodes, output_size)

        return out
