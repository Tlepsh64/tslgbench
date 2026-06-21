"""
Graph visualization utilities for comparing ground truth, initial, and learned graphs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


def edge_index_to_adjacency(edge_index, n_nodes):
    """Convert edge_index (2, E) to adjacency matrix (n_nodes, n_nodes).

    Edges involving node indices >= n_nodes are skipped (e.g., dummy nodes).
    """
    adj = np.zeros((n_nodes, n_nodes))
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        # Skip edges involving nodes outside the valid range (e.g., dummy nodes)
        if src < n_nodes and dst < n_nodes:
            adj[src, dst] = 1
            adj[dst, src] = 1
    return adj


def adjacency_to_edge_list(adj, threshold=0.5):
    """Convert adjacency matrix to list of edges. For soft adjacency, apply threshold."""
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    edges = []
    n_nodes = adj.shape[0]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Upper triangular to avoid duplicates
            if adj[i, j] > threshold:
                edges.append((i, j))
    return edges


def plot_graph(adj, node_positions, ax, title="Graph", node_size=300,
               edge_color='black', node_color='lightblue', alpha=1.0,
               highlight_edges=None, highlight_color='red'):
    """
    Plot a graph with fixed node positions and node IDs.

    Args:
        adj: Adjacency matrix (n_nodes, n_nodes) - can be binary or soft
        node_positions: Node positions array (n_nodes, 2)
        ax: Matplotlib axis to plot on
        title: Title for the plot
        node_size: Size of nodes
        edge_color: Color for edges
        node_color: Color for nodes
        alpha: Transparency for edges
        highlight_edges: List of (i, j) tuples to highlight
        highlight_color: Color for highlighted edges
    """
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    if isinstance(node_positions, torch.Tensor):
        node_positions = node_positions.cpu().numpy()

    n_nodes = adj.shape[0]

    # Create networkx graph
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i, pos=node_positions[i])

    # Add edges
    edges = adjacency_to_edge_list(adj, threshold=0.5)
    G.add_edges_from(edges)

    # Get positions dict for networkx
    pos = {i: node_positions[i] for i in range(n_nodes)}

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')

    # Draw regular edges
    if highlight_edges:
        highlight_set = set(highlight_edges) | set((j, i) for i, j in highlight_edges)
        regular_edges = [(u, v) for u, v in G.edges() if (u, v) not in highlight_set and (v, u) not in highlight_set]
        highlight_edge_list = [(u, v) for u, v in G.edges() if (u, v) in highlight_set or (v, u) in highlight_set]

        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=regular_edges,
                               edge_color=edge_color, alpha=alpha, width=1.5)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=highlight_edge_list,
                               edge_color=highlight_color, alpha=1.0, width=3.0)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, alpha=alpha, width=1.5)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')


def visualize_graphs(ground_truth_adj, learned_adj, node_positions, save_path,
                     initial_adj=None, hidden_edge=None):
    """
    Create a comparison visualization of ground truth, initial (if any), and learned graphs.

    Args:
        ground_truth_adj: Ground truth adjacency matrix (n_nodes, n_nodes)
        learned_adj: Learned adjacency matrix (n_nodes, n_nodes) - from predictor.test_graphs[0]
        node_positions: Node positions array (n_nodes, 2)
        save_path: Path to save the figure
        initial_adj: Initial/prior adjacency matrix (optional, only for learn_one_edge)
        hidden_edge: Tuple (i, j) of hidden edge to highlight (optional, only for learn_one_edge)
    """
    n_plots = 3 if initial_adj is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))

    # Handle axes indexing
    if n_plots == 2:
        ax_gt = axes[0]
        ax_learned = axes[1]
        ax_initial = None
    else:
        ax_gt = axes[0]
        ax_initial = axes[1]
        ax_learned = axes[2]

    # Plot ground truth
    # Handle both single edge (tuple) and multiple edges (list of tuples)
    if hidden_edge is None:
        highlight = None
    elif isinstance(hidden_edge, list):
        highlight = hidden_edge  # Already a list of edges
    else:
        highlight = [hidden_edge]  # Single edge, wrap in list
    plot_graph(ground_truth_adj, node_positions, ax_gt,
               title="Ground Truth Graph",
               node_color='lightgreen',
               highlight_edges=highlight,
               highlight_color='red')

    # Plot initial graph if provided (only for learn_one_edge)
    if initial_adj is not None and ax_initial is not None:
        plot_graph(initial_adj, node_positions, ax_initial,
                   title="Initial Graph (Given to Framework)",
                   node_color='lightyellow',
                   highlight_edges=highlight,
                   highlight_color='orange')

    # Plot learned graph (from first MC sample)
    plot_graph(learned_adj, node_positions, ax_learned,
               title="Test Graph (First MC Sample)",
               node_color='lightblue',
               highlight_edges=highlight,
               highlight_color='blue')

    plt.tight_layout()

    # Ensure save directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved graph comparison to {save_path}")


def visualize_edge_comparison(ground_truth_adj, learned_adj, node_positions, save_path,
                              hidden_edge=None):
    """
    Create a detailed edge comparison visualization showing correct, missing, and extra edges.

    Args:
        ground_truth_adj: Ground truth adjacency matrix
        learned_adj: Learned adjacency matrix
        node_positions: Node positions array (n_nodes, 2)
        save_path: Path to save the figure
        hidden_edge: Tuple (i, j) of hidden edge to highlight
    """
    if isinstance(ground_truth_adj, torch.Tensor):
        ground_truth_adj = ground_truth_adj.cpu().numpy()
    if isinstance(learned_adj, torch.Tensor):
        learned_adj = learned_adj.cpu().numpy()
    if isinstance(node_positions, torch.Tensor):
        node_positions = node_positions.cpu().numpy()

    n_nodes = ground_truth_adj.shape[0]

    # Compute edge sets
    gt_edges = set(adjacency_to_edge_list(ground_truth_adj))
    learned_edges = set(adjacency_to_edge_list(learned_adj))

    correct_edges = gt_edges & learned_edges  # True positives
    missing_edges = gt_edges - learned_edges  # False negatives
    extra_edges = learned_edges - gt_edges    # False positives

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Create networkx graph with all possible edges for visualization
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i, pos=node_positions[i])

    pos = {i: node_positions[i] for i in range(n_nodes)}

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=400, node_color='lightgray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')

    # Draw edges with different colors
    # Correct edges (green)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(correct_edges),
                           edge_color='green', alpha=0.8, width=2.0, label='Correct')
    # Missing edges (red, dashed)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(missing_edges),
                           edge_color='red', alpha=0.8, width=2.0, style='dashed', label='Missing')
    # Extra edges (orange)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(extra_edges),
                           edge_color='orange', alpha=0.8, width=2.0, label='Extra')

    # Highlight hidden edge(s) if specified
    if hidden_edge:
        # Handle both single edge (tuple) and multiple edges (list of tuples)
        if isinstance(hidden_edge, list):
            hidden_edges = hidden_edge
        else:
            hidden_edges = [hidden_edge]

        # Count learned vs not learned
        learned_count = 0
        not_learned_count = 0
        for edge in hidden_edges:
            if edge in correct_edges or (edge[1], edge[0]) in correct_edges:
                learned_count += 1
            elif edge in missing_edges or (edge[1], edge[0]) in missing_edges:
                not_learned_count += 1

        if len(hidden_edges) == 1:
            # Single edge: show detailed status
            edge = hidden_edges[0]
            if edge in correct_edges or (edge[1], edge[0]) in correct_edges:
                status = "LEARNED CORRECTLY"
                color = 'blue'
            elif edge in missing_edges or (edge[1], edge[0]) in missing_edges:
                status = "NOT LEARNED"
                color = 'purple'
            else:
                status = "N/A"
                color = 'gray'
            ax.annotate(f'Hidden edge {edge}: {status}',
                        xy=(0.5, 0.02), xycoords='axes fraction',
                        ha='center', fontsize=11, fontweight='bold', color=color)
        else:
            # Multiple edges: show summary
            if not_learned_count == 0:
                color = 'blue'
            elif learned_count == 0:
                color = 'purple'
            else:
                color = 'orange'
            ax.annotate(f'Hidden edges: {learned_count}/{len(hidden_edges)} learned correctly',
                        xy=(0.5, 0.02), xycoords='axes fraction',
                        ha='center', fontsize=11, fontweight='bold', color=color)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, label=f'Correct ({len(correct_edges)})'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='dashed', label=f'Missing ({len(missing_edges)})'),
        Line2D([0], [0], color='orange', linewidth=2, label=f'Extra ({len(extra_edges)})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_title("Edge Comparison: Ground Truth vs Learned", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved edge comparison to {save_path}")


def extract_learned_adjacency(predictor, datamodule, device='cuda', n_nodes=None):
    """
    Extract the learned adjacency matrix from a trained predictor.

    Args:
        predictor: Trained predictor (SFGraphPredictor, IMLEGraphPredictor, etc.)
        datamodule: Data module with test data
        device: Device to run inference on
        n_nodes: Number of real nodes (excluding dummy nodes). If None, uses all nodes.

    Returns:
        learned_adj: Learned adjacency matrix (n_nodes, n_nodes)
    """
    predictor.eval()
    predictor = predictor.to(device)

    # Get a batch from test loader
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))

    # Move batch to device
    batch = batch.to(device)

    with torch.no_grad():
        # Get the graph module's output
        x = batch.x  # Input features
        connectivity = predictor.graph_module(x)

        # Determine n_nodes if not provided
        total_nodes = predictor.graph_module.edge_scorer.num_nodes
        if n_nodes is None:
            n_nodes = total_nodes

        # Use mean graph for evaluation (mode)
        if 'mean_graph' in connectivity:
            mean_edge_index = connectivity['mean_graph']['edge_index']
            # Convert to adjacency matrix (edges outside n_nodes are filtered)
            learned_adj = edge_index_to_adjacency(mean_edge_index, n_nodes)
        elif 'logits' in connectivity:
            # Use sigmoid of logits as soft adjacency
            logits = connectivity['logits']
            if logits.dim() == 3:
                logits = logits[0]  # Take first batch element
            learned_adj = torch.sigmoid(logits).cpu().numpy()
            # Slice to n_nodes if needed
            if learned_adj.shape[0] > n_nodes:
                learned_adj = learned_adj[:n_nodes, :n_nodes]
        else:
            # Fallback to sampled adjacency
            adj = connectivity.get('adj', None)
            if adj is not None:
                if adj.dim() == 3:
                    adj = adj[0]
                learned_adj = adj.cpu().numpy()
                # Slice to n_nodes if needed
                if learned_adj.shape[0] > n_nodes:
                    learned_adj = learned_adj[:n_nodes, :n_nodes]
            else:
                raise ValueError("Could not extract learned adjacency from predictor")

    return learned_adj
