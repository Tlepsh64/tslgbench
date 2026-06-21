# -*- coding: utf-8 -*-
"""
Batched Kruskal MST solver used as the AIMLE MAP solver, adapted from
torch-adaptive-imle (https://github.com/EdinburghNLP/torch-adaptive-imle),
MIT License, Copyright (c) Minervini, Franceschi, Niepert.
"""

import torch
from torch import Tensor


def kruskal_mst_batched(scores: Tensor) -> Tensor:
    """
    Compute Maximum Spanning Tree using Kruskal's algorithm for batched adjacency matrices.

    Args:
        scores: [batch_size, n_nodes, n_nodes] - edge weights (higher = more likely to be in MST)

    Returns:
        adj: [batch_size, n_nodes, n_nodes] - binary adjacency matrix of MST
    """
    batch_size, n_nodes, _ = scores.shape
    device = scores.device

    # Initialize output adjacency matrices
    adj = torch.zeros_like(scores)

    for b in range(batch_size):
        # Get upper triangular indices (undirected graph)
        triu_indices = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)
        edge_weights = scores[b, triu_indices[0], triu_indices[1]]

        # Sort edges by weight in descending order (for maximum spanning tree)
        sorted_indices = torch.argsort(edge_weights, descending=True)

        # Union-Find data structure
        parent = list(range(n_nodes))
        rank = [0] * n_nodes

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        # Kruskal's algorithm
        edges_added = 0
        for idx in sorted_indices:
            if edges_added >= n_nodes - 1:
                break
            i = triu_indices[0][idx].item()
            j = triu_indices[1][idx].item()
            if union(i, j):
                # Add edge in both directions (undirected)
                adj[b, i, j] = 1.0
                adj[b, j, i] = 1.0
                edges_added += 1

    return adj
