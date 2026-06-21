"""
TTS-GAT (Time-Then-Space with Graph Attention) Model for spatiotemporal forecasting.

Same architecture as TTSModel but replaces GCN decoder with GAT decoder:
- GRU encoder (multi-layer) to process temporal dimension
- GAT decoder (multi-layer Graph Attention) for spatial processing
"""

import torch
import torch.nn as nn
from tsl.nn.models import BaseModel
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers.graph_convs.gat_conv import GATConv
from tsl.nn.blocks.decoders.mlp_decoder import MLPDecoder
from tsl.nn.utils import get_functional_activation


class GATDecoder(nn.Module):
    """GAT decoder for multistep forecasting.

    Applies multiple graph attention layers followed by a feed-forward layer
    and a linear readout. If the input representation has a temporal dimension,
    this model will take as input the representation corresponding to the
    last step.

    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size (must be divisible by heads).
        output_size (int): Output size.
        horizon (int): Number of time steps in the prediction horizon.
        n_layers (int): Number of GAT layers in the decoder.
        heads (int): Number of attention heads.
        activation (str): Activation function.
        dropout (float): Dropout probability.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 horizon: int = 1,
                 n_layers: int = 1,
                 heads: int = 4,
                 activation: str = 'relu',
                 dropout: float = 0.):
        super().__init__()
        graph_convs = []
        for i in range(n_layers):
            in_size = input_size if i == 0 else hidden_size
            graph_convs.append(
                GATConv(in_channels=in_size,
                        out_channels=hidden_size,
                        heads=heads,
                        concat=True,
                        dropout=dropout)
            )
        self.convs = nn.ModuleList(graph_convs)
        self.activation = get_functional_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  output_size=output_size,
                                  activation=activation,
                                  horizon=horizon)

    def forward(self, h, edge_index, edge_weight=None):
        if h.dim() == 4:
            h = h[:, -1]
        for conv in self.convs:
            h, _ = conv(h, edge_index)
            h = self.dropout(self.activation(h))
        return self.readout(h)


class TTSGATModel(BaseModel):
    """
    Time-Then-Space model with GAT decoder.

    Same as TTSModel but uses Graph Attention Networks instead of GCN
    for the spatial decoder.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 output_size: int = 1,
                 horizon: int = 1,
                 n_nodes: int = None,
                 enc_layers: int = 2,
                 gcn_layers: int = 2,
                 heads: int = 4,
                 dropout: float = 0.0,
                 activation: str = 'relu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.horizon = horizon
        self.n_nodes = n_nodes

        # GRU encoder: processes temporal dimension
        self.encoder = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=enc_layers,
            return_only_last_state=True,
            cell='gru',
            dropout=dropout
        )

        # GAT decoder: processes spatial dimension using learned graph
        self.decoder = GATDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size * horizon,
            n_layers=gcn_layers,
            heads=heads,
            activation=activation,
            dropout=dropout
        )

    def forward(self, x, edge_index, edge_weight=None, disjoint=False, **kwargs):
        batch_size = x.size(0)
        n_nodes = x.size(2)

        # Encode temporal information with GRU
        h = self.encoder(x)

        n_nodes_with_dummy = n_nodes

        # Decode with GAT layers
        if disjoint:
            h = h.view(batch_size * n_nodes_with_dummy, self.hidden_size)

            for conv in self.decoder.convs:
                h, _ = conv(h, edge_index)
                h = self.decoder.dropout(self.decoder.activation(h))

            h = h.view(batch_size, n_nodes_with_dummy, -1)
            out = self.decoder.readout(h)
        else:
            out = self.decoder(h, edge_index)

        # Reshape output to (batch, horizon, nodes, output_size)
        out = out.view(batch_size, n_nodes, self.horizon, self.output_size)
        out = out.permute(0, 2, 1, 3)

        return out
