"""
TTS (Time-Then-Space) Model for spatiotemporal forecasting.

Architecture follows Cini et al. (2023) Section 8.3.1:
- GRU encoder (2 hidden layers) to process temporal dimension
- GCN decoder (2 graph convolutional layers) for spatial processing
- Hidden size: 64, Input window: 24 steps
"""

import torch
import torch.nn as nn
from tsl.nn.models import BaseModel
from tsl.nn.blocks.encoders import RNN
from tsl.nn.blocks.decoders import GCNDecoder


class TTSModel(BaseModel):
    """
    Time-Then-Space model: GRU encoder followed by GCN decoder.

    First processes temporal information with GRU, then applies
    graph convolutions for spatial message passing.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 output_size: int = 1,
                 horizon: int = 1,
                 n_nodes: int = None,
                 enc_layers: int = 2,
                 gcn_layers: int = 2,
                 dropout: float = 0.0,
                 activation: str = 'relu'):
        """
        Args:
            input_size: Number of input features per node per timestep
            hidden_size: Size of hidden representations (default: 64)
            output_size: Number of output features per node (default: 1)
            horizon: Number of steps to forecast (default: 1)
            n_nodes: Number of real nodes in the graph (optional)
            enc_layers: Number of GRU layers in encoder (default: 2)
            gcn_layers: Number of GCN layers in decoder (default: 2)
            dropout: Dropout rate (default: 0.0)
            activation: Activation function (default: 'relu')
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

        # GCN decoder: processes spatial dimension using learned graph
        self.decoder = GCNDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size * horizon,
            n_layers=gcn_layers,
            activation=activation,
            dropout=dropout
        )

        # GCN-style normalization required to prevent division by zero. 
        # Default norm is "mean"(D^-1 A) which may create issues(division by 0) when any node has zero in-degrees.
        #for conv in self.decoder.convs:
        #    conv.norm = 'gcn'

    def forward(self, x, edge_index, edge_weight=None, disjoint=False, **kwargs):
        """
        Forward pass of TTS model.

        Args:
            x: Input tensor of shape (batch, time, nodes, features)
            edge_index: Graph connectivity in COO format (2, num_edges)
            edge_weight: Optional edge weights (num_edges,)
            disjoint: If True, edge_index uses disjoint batching where each
                      batch has unique node IDs (batch i uses nodes i*n to (i+1)*n-1)
            **kwargs: Additional arguments (e.g., adj)

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

        n_nodes_with_dummy = n_nodes

        # Decode with GCN layers
        if disjoint:
            # For disjoint batching, edge_index uses unique node IDs per batch
            # (batch i uses nodes i*n to (i+1)*n-1)
            # Flatten h to 2D for GCN convolutions, then unflatten for readout
            h = h.view(batch_size * n_nodes_with_dummy, self.hidden_size)

            # Run through decoder's GCN convolutions with 2D input
            for conv in self.decoder.convs:
                h = self.decoder.dropout(self.decoder.activation(conv(h, edge_index, edge_weight)))

            # Unflatten back to 3D for readout layer
            h = h.view(batch_size, n_nodes_with_dummy, -1)

            # Apply readout layer (expects 3D input)
            out = self.decoder.readout(h)
        else:
            # Standard batching: edge_index is shared across batches
            out = self.decoder(h, edge_index, edge_weight)

        # Reshape output to (batch, horizon, nodes, output_size)
        # GCNDecoder outputs (batch, nodes, horizon * output_size)
        out = out.view(batch_size, n_nodes, self.horizon, self.output_size)
        out = out.permute(0, 2, 1, 3)  # (batch, horizon, nodes, output_size)

        return out
