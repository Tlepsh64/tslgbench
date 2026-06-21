"""
SatorrasForecaster: Forecasting model based on Satorras et al. (2022)
"Multivariate Time Series Forecasting with Latent Graph Inference"

Adapted for sparse graph learning following Cini et al. (2022) Section 8.3.2:
the gating mechanism is removed from the GNN since graph structure is
provided externally by the graph learning module (BES/SNS sampler).

Architecture:
- Configurable encoder (GRU, LSTM, or MLP) to process temporal dimension
- UngatedGraphNetwork layers for spatial message passing
- MLP decoder with residual connection
- Optional learnable node ID embeddings (c_i from Satorras Section 4)
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj

from tsl.nn.models import BaseModel
from tsl.nn.blocks.encoders import RNN
from tsl.nn.utils import get_layer_activation


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class UngatedGraphNetwork(MessagePassing):
    """GatedGraphNetwork (Satorras et al.) with gating removed.

    Following Cini et al. (2022) Section 8.3.2, the attention gate phi_alpha
    is removed since graph sparsity is handled externally by the graph
    learning module.

    The layer computes:
        m_ij = phi_e([h_i, h_j])          (edge function, 2-layer MLP)
        m_i  = sum_{j in N(i)} m_ij       (aggregation)
        h_i' = phi_h([h_i, m_i]) + skip(h_i)  (node update + residual)
    """

    def __init__(self, input_size: int, output_size: int,
                 activation: str = 'silu',
                 parametrized_skip_conn: bool = False):
        super().__init__(aggr="add", node_dim=-2)

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * input_size, output_size // 2),
            get_layer_activation(activation)(),
            nn.Linear(output_size // 2, output_size),
            get_layer_activation(activation)(),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            get_layer_activation(activation)(),
            nn.Linear(output_size, output_size),
        )

        if (input_size != output_size) or parametrized_skip_conn:
            self.skip_conn = nn.Linear(input_size, output_size)
        else:
            self.skip_conn = nn.Identity()

    def forward(self, x: Tensor, edge_index: Adj):
        out = self.propagate(edge_index, x=x)
        out = self.update_mlp(torch.cat([out, x], -1)) + self.skip_conn(x)
        return out

    def message(self, x_i: Tensor, x_j: Tensor):
        return self.msg_mlp(torch.cat([x_i, x_j], -1))


class MLPResBlock(nn.Module):
    """Residual MLP block (MLP_res from Satorras et al. Appendix A.2).

    Input -> Linear -> Activation -> Linear -> + Input -> Output
    """

    def __init__(self, size: int, activation: str = 'silu'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(size, size),
            get_layer_activation(activation)(),
            nn.Linear(size, size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(x)


class MLPEncoder(nn.Module):
    """MLP encoder for temporal processing (Satorras et al. Appendix A.2).

    Flattens the input window into a feature vector and processes it through
    a linear projection followed by residual MLP blocks.

    For METR-LA / PEMS-BAY, the encoder is:
        x_{i,t0:t} -> Linear(in_dim, nf) -> MLP_res -> MLP_res -> z_i
    """

    def __init__(self, input_size: int, window: int, hidden_size: int,
                 n_layers: int = 2, activation: str = 'silu'):
        super().__init__()
        self.proj = nn.Linear(input_size * window, hidden_size)
        self.blocks = nn.ModuleList([
            MLPResBlock(hidden_size, activation) for _ in range(n_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, time, nodes, features)
        b, t, n, f = x.shape
        # Flatten temporal dim: (batch, nodes, time * features)
        x = x.permute(0, 2, 1, 3).reshape(b, n, t * f)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return x  # (batch, nodes, hidden_size)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SatorrasForecaster(BaseModel):
    """Forecasting model based on Satorras et al. (2022), adapted for sparse
    graph learning.

    The encoder processes each node's time series independently. The GNN
    aggregation module propagates information across nodes using the graph
    provided by the graph learning module. The decoder produces per-node
    forecasts from the updated embeddings.

    Args:
        input_size (int): Number of input features per node per timestep.
        hidden_size (int): Hidden representation size (nf in the paper).
        output_size (int): Number of output features per node.
        horizon (int): Number of future steps to forecast.
        n_nodes (int): Number of nodes (needed for node ID embeddings).
        window (int): Input window size (needed for MLP encoder).
        encoder_type (str): Encoder type: ``'gru'``, ``'lstm'``, or ``'mlp'``.
        enc_layers (int): Number of encoder layers.
        gnn_layers (int): Number of GNN layers in the aggregation module.
        dropout (float): Dropout probability.
        activation (str): Activation function (default: ``'silu'`` / Swish).
        use_node_id (bool): If True, concatenate learnable node ID embeddings
            to the encoder output (c_i from Section 4 of the paper).
        node_emb_size (int, optional): Dimension of node ID embeddings.
            Defaults to ``hidden_size`` if not specified.
        use_gate (bool): If True, use the original gated GNN from Satorras.
            If False (default), use the ungated variant from Cini et al.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 output_size: int = 1,
                 horizon: int = 1,
                 n_nodes: int = None,
                 window: int = 12,
                 encoder_type: str = 'mlp',
                 enc_layers: int = 2,
                 gnn_layers: int = 2,
                 dropout: float = 0.0,
                 activation: str = 'silu',
                 use_node_id: bool = True,
                 node_emb_size: int = None,
                 use_gate: bool = False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.encoder_type = encoder_type
        self.use_node_id = use_node_id and (n_nodes is not None)
        self.use_gate = use_gate

        # ---- Encoder ----
        if encoder_type == 'mlp':
            self.encoder = MLPEncoder(
                input_size=input_size,
                window=window,
                hidden_size=hidden_size,
                n_layers=enc_layers,
                activation=activation,
            )
        elif encoder_type in ('gru', 'lstm'):
            self.encoder = RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                n_layers=enc_layers,
                return_only_last_state=True,
                cell=encoder_type,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}'. "
                f"Choose from 'gru', 'lstm', 'mlp'."
            )

        # ---- Node ID embeddings (optional) ----
        gnn_input_size = hidden_size
        if self.use_node_id:
            self._node_emb_size = node_emb_size or hidden_size
            self.node_emb = nn.Embedding(n_nodes, self._node_emb_size)
            gnn_input_size = hidden_size + self._node_emb_size

        # ---- GNN aggregation module ----
        from tsl.nn.layers.graph_convs.gated_gn import GatedGraphNetwork
        gnn_cls = GatedGraphNetwork if use_gate else UngatedGraphNetwork

        gnns = []
        for i in range(gnn_layers):
            in_size = gnn_input_size if i == 0 else hidden_size
            gnns.append(gnn_cls(in_size, hidden_size, activation=activation))
        self.gnns = nn.ModuleList(gnns)
        self.gnn_dropout = nn.Dropout(dropout)

        # ---- Decoder (MLP, Satorras Appendix A.2) ----
        self.decoder = nn.Sequential(
            MLPResBlock(hidden_size, activation),
            nn.Linear(hidden_size, output_size * horizon),
        )

    def forward(self, x, edge_index, edge_weight=None,
                disjoint=False, **kwargs):
        """
        Args:
            x: Input tensor (batch, time, nodes, features).
            edge_index: Graph connectivity in COO format (2, num_edges).
            edge_weight: Unused (kept for interface compatibility).
            disjoint: If True, edge_index uses disjoint per-batch node IDs.

        Returns:
            Predictions of shape (batch, horizon, nodes, output_size).
        """
        batch_size = x.size(0)
        n_nodes = x.size(2)

        # --- Encode temporal information ---
        h = self.encoder(x)  # (batch, nodes, hidden_size)

        # --- Concatenate node ID embeddings ---
        if self.use_node_id:
            node_ids = torch.arange(n_nodes, device=x.device)
            emb = self.node_emb(node_ids)                       # (nodes, emb)
            emb = emb.unsqueeze(0).expand(batch_size, -1, -1)   # (batch, nodes, emb)
            h = torch.cat([h, emb], dim=-1)  # (batch, nodes, hidden_size + emb)

        # --- GNN spatial processing ---
        if disjoint:
            h = h.view(batch_size * n_nodes, -1)
            for gnn in self.gnns:
                h = self.gnn_dropout(gnn(h, edge_index))
            h = h.view(batch_size, n_nodes, -1)
        else:
            for gnn in self.gnns:
                h = self.gnn_dropout(gnn(h, edge_index))

        # --- Decode ---
        out = self.decoder(h)  # (batch, nodes, output_size * horizon)
        out = out.view(batch_size, n_nodes, self.horizon, self.output_size)
        out = out.permute(0, 2, 1, 3)  # (batch, horizon, nodes, output_size)

        return out
