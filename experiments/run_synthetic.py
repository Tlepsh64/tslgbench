import os
from tsl.experiment import Experiment
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
import tsl
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
import torch
import lib
from lib.predictors.latent_graph_predictor import LatentGraphPredictor, SFGraphPredictor, IMLEGraphPredictor, AIMLEGraphPredictor
from lib.nn.graph_module import GraphModule, IMLEGraphModule, AIMLEGraphModule
from lib.nn.tts_model import TTSModel
from lib.datasets.graph_polynomial_var import GraphPolyVARDataset, GraphPolyVARDatasetTriCommunity, GraphPolyVARDatasetErdosRenyi, GraphPolyVARDatasetSBM, GraphPolyVARDatasetBarabasiAlbert, GraphPolyVARDatasetLadder, GraphPolyVARDatasetGrid, GraphPolyVARDatasetRandomTree

import time
import psutil
import gc
import json
import numpy as np
from lib.utils.visualization import (
    visualize_graphs,
    visualize_edge_comparison,
    edge_index_to_adjacency
)

def get_gpu_memory_usage():
    """Get GPU memory usage if available"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
    return 0.0

def get_dataset(cfg):
    dataset_name = cfg.dataset.name

    # Get common dataset parameters from config with defaults
    node_num = getattr(cfg.dataset, 'node_num', 30)
    time_series_noise = getattr(cfg.dataset, 'time_series_noise', 0.4)

    if dataset_name == 'gpolyvar':
        T = 30000
        # Calculate communities based on node count (6 nodes per community)
        communities = node_num // 6
        # connectivity is only used by gpolyvar (TriCommunity graph)
        connectivity = getattr(cfg.dataset, 'connectivity', 'line')
        data_path = os.path.join(lib.config['data_dir'], f"gpvar-T{T}_{connectivity}-c{communities}-n{node_num}-noise{time_series_noise}")

        dataset = GraphPolyVARDatasetTriCommunity(coefs=torch.tensor([[5, 2], [-4, 6], [-1, 0]], dtype=torch.float32),
                                      sigma_noise=time_series_noise,
                                      communities=communities, connectivity=connectivity)
        dataset.generate_data(T=T)
        dataset.dump_dataset(path=data_path)

    elif dataset_name == 'gpolyvar_erdos_renyi':
        T = 30000
        p = 0.15
        seed = None # Set for reproducability, but feel free to remove.
        data_path = os.path.join(lib.config['data_dir'], f"gpvar-T{T}_erdos_n{node_num}_p{p}_noise{time_series_noise}")

        dataset = GraphPolyVARDatasetErdosRenyi(
            coefs=torch.tensor([[5, 2], [-4, 6], [-1, 0]], dtype=torch.float32),
            sigma_noise=time_series_noise,
            n=node_num, p=p, seed=seed
        )
        dataset.generate_data(T=T)
        dataset.dump_dataset(path=data_path)

    elif dataset_name == 'gpolyvar_sbm':
        T = 30000
        # Calculate number of blocks based on node count (6 nodes per block)
        num_blocks = node_num // 6
        block_sizes = [6] * num_blocks
        # High within-community probability, low between-community probability
        edge_probs = [[0.6 if i == j else 0.05 for j in range(num_blocks)] for i in range(num_blocks)]
        seed = None
        data_path = os.path.join(lib.config['data_dir'], f"gpvar-T{T}_sbm_n{node_num}_blocks{num_blocks}_noise{time_series_noise}")

        dataset = GraphPolyVARDatasetSBM(
            coefs=torch.tensor([[5, 2], [-4, 6], [-1, 0]], dtype=torch.float32),
            sigma_noise=time_series_noise,
            block_sizes=block_sizes,
            edge_probs=edge_probs,
            seed=seed
        )
        dataset.generate_data(T=T)
        dataset.dump_dataset(path=data_path)

    elif dataset_name == 'gpolyvar_barabasi_albert':
        T = 30000
        m = 2  # Number of edges to attach from a new node to existing nodes
        seed = 42
        data_path = os.path.join(lib.config['data_dir'], f"gpvar-T{T}_ba_n{node_num}_m{m}_noise{time_series_noise}")

        dataset = GraphPolyVARDatasetBarabasiAlbert(
            coefs=torch.tensor([[5, 2], [-4, 6], [-1, 0]], dtype=torch.float32),
            sigma_noise=time_series_noise,
            n=node_num,
            m=m,
            seed=seed
        )
        dataset.generate_data(T=T)
        dataset.dump_dataset(path=data_path)

    elif dataset_name == 'gpolyvar_ladder':
        T = 30000
        n = node_num // 2  # ladder_graph(n) produces 2n nodes
        seed = None
        data_path = os.path.join(lib.config['data_dir'], f"gpvar-T{T}_ladder_n{node_num}_noise{time_series_noise}")

        dataset = GraphPolyVARDatasetLadder(
            coefs=torch.tensor([[5, 2], [-4, 6], [-1, 0]], dtype=torch.float32),
            sigma_noise=time_series_noise,
            n=n,
            seed=seed
        )
        dataset.generate_data(T=T)
        dataset.dump_dataset(path=data_path)

    elif dataset_name == 'gpolyvar_grid':
        T = 30000
        # Find closest factorization of node_num into rows x cols
        import math
        sqrt_n = int(math.isqrt(node_num))
        while node_num % sqrt_n != 0:
            sqrt_n -= 1
        rows, cols = sqrt_n, node_num // sqrt_n
        seed = None
        data_path = os.path.join(lib.config['data_dir'], f"gpvar-T{T}_grid_{rows}x{cols}_noise{time_series_noise}")

        dataset = GraphPolyVARDatasetGrid(
            coefs=torch.tensor([[5, 2], [-4, 6], [-1, 0]], dtype=torch.float32),
            sigma_noise=time_series_noise,
            rows=rows,
            cols=cols,
            seed=seed
        )
        dataset.generate_data(T=T)
        dataset.dump_dataset(path=data_path)

    elif dataset_name == 'gpolyvar_random_tree':
        T = 30000
        seed = 42
        data_path = os.path.join(lib.config['data_dir'], f"gpvar-T{T}_tree_n{node_num}_noise{time_series_noise}")

        dataset = GraphPolyVARDatasetRandomTree(
            coefs=torch.tensor([[5, 2], [-4, 6], [-1, 0]], dtype=torch.float32),
            sigma_noise=time_series_noise,
            n=node_num,
            seed=seed
        )
        dataset.generate_data(T=T)
        dataset.dump_dataset(path=data_path)

    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset

def create_learn_one_edge_prior(dataset, hidden_edge_idx=0, seed=42, dummy_nodes=0):
    """
    Create fixed adjacency and learnable mask for the learn_one_edge experiment.

    Only the hidden edge is sampled; all other edges are deterministically fixed
    to their true values (no sampling uncertainty for known edges).

    Args:
        dataset: The GraphPolyVARDataset containing the true graph
        hidden_edge_idx: Index of which edge to hide (0 = first internal edge in first community)
        seed: Random seed for selecting which edge to hide
        dummy_nodes: Number of dummy nodes added by the graph module

    Returns:
        init_logits: (n_nodes + dummy_nodes, n_nodes + dummy_nodes) tensor with initial logits
        learnable_mask: (n_nodes + dummy_nodes, n_nodes + dummy_nodes) boolean tensor (True = learnable)
        fixed_adj: (n_nodes + dummy_nodes, n_nodes + dummy_nodes) tensor with true adjacency values
        fixed_mask: (n_nodes + dummy_nodes, n_nodes + dummy_nodes) boolean tensor (True = use fixed value)
        hidden_edge: tuple (i, j) of the hidden edge indices
    """
    import numpy as np

    # Get the true edge_index from the graph
    edge_index = dataset.G.edge_index  # shape (2, num_edges)
    n_nodes = dataset.G.num_nodes
    total_nodes = n_nodes + dummy_nodes

    # Create true adjacency matrix (including space for dummy nodes)
    true_adj = torch.zeros(total_nodes, total_nodes)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        true_adj[src, dst] = 1.0
        true_adj[dst, src] = 1.0  # Ensure symmetric
    # Dummy nodes have no edges (remain 0)

    # Find all edges in the first community (nodes 0-5 for tricom)
    community_edges = []
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src < 6 and dst < 6 and src < dst:  # First community, no duplicates
            community_edges.append((src, dst))

    if len(community_edges) == 0:
        raise ValueError("No internal edges found in first community")

    # Select edge to hide
    np.random.seed(seed)
    edge_to_hide_idx = hidden_edge_idx % len(community_edges)
    hidden_edge = community_edges[edge_to_hide_idx]
    i, j = hidden_edge

    print(f"[learn_one_edge] Hiding edge ({i}, {j}) from true graph")
    print(f"[learn_one_edge] Total edges in graph: {edge_index.shape[1] // 2}")
    print(f"[learn_one_edge] Edges in first community: {len(community_edges)}")
    print(f"[learn_one_edge] Total nodes (with {dummy_nodes} dummy): {total_nodes}")

    # Create init_logits: only the hidden edge position has a learnable value
    # Other positions don't matter since they'll be overridden by fixed_adj
    init_logits = torch.zeros(total_nodes, total_nodes)
    init_logits[i, j] = 0.0  # Start at neutral (50/50 probability)
    init_logits[j, i] = 0.0

    # Learnable mask: only hidden edge can be updated during training
    learnable_mask = torch.zeros(total_nodes, total_nodes, dtype=torch.bool)
    learnable_mask[i, j] = True
    learnable_mask[j, i] = True

    # Fixed adjacency: the true graph values (dummy nodes have no edges)
    fixed_adj = true_adj.clone()

    # Fixed mask: everything EXCEPT the hidden edge is fixed (not sampled)
    fixed_mask = torch.ones(total_nodes, total_nodes, dtype=torch.bool)
    fixed_mask[i, j] = False  # Hidden edge is sampled, not fixed
    fixed_mask[j, i] = False

    return init_logits, learnable_mask, fixed_adj, fixed_mask, hidden_edge

def create_learn_multiple_edges_prior(dataset, num_hidden_edges, seed=42, dummy_nodes=0):
    """
    Create prior for learning edges spread across communities.

    Strategy:
    - First, hide edge (0,1) in each community (up to num_communities edges)
    - Then, hide edge (4,5) in each community
    - Then, hide edge (1,2) in each community
    - etc.

    Example for 30 nodes (5 communities), num_hidden_edges=6:
      (0,1), (6,7), (12,13), (18,19), (24,25),  # edge (0,1) in all 5 communities
      (4,5)                                       # edge (4,5) in community 0

    Example for 30 nodes, num_hidden_edges=7:
      (0,1), (6,7), (12,13), (18,19), (24,25),  # edge (0,1) in all 5 communities
      (4,5), (10,11)                             # edge (4,5) in communities 0 and 1

    Returns:
        init_logits: Initial logit values for the edge scorer
        learnable_mask: Boolean mask for which logits can be updated
        fixed_adj: The adjacency values to use for fixed (non-sampled) edges
        fixed_mask: Boolean mask for which edges are fixed vs sampled
        hidden_edges: List of (i, j) tuples of hidden edges
    """
    edge_index = dataset.G.edge_index
    n_nodes = dataset.G.num_nodes
    total_nodes = n_nodes + dummy_nodes
    num_communities = n_nodes // 6

    # Create true adjacency matrix (including space for dummy nodes)
    true_adj = torch.zeros(total_nodes, total_nodes)
    for k in range(edge_index.shape[1]):
        src, dst = edge_index[0, k], edge_index[1, k]
        true_adj[src, dst] = 1.0
        true_adj[dst, src] = 1.0

    # Edge types to cycle through (local indices within each community)
    # Priority order: spread edges across communities first
    edge_types = [
        (0, 1),  # First priority - bottom-left to neighbor above
        (4, 5),  # Second priority - opposite corner
        (1, 2),  # Third priority
        (3, 4),
        (1, 3),
        (2, 4),
        (0, 3),
        (1, 4),
        (3, 5),
    ]

    # Select edges by cycling through communities for each edge type
    hidden_edges = []
    edge_type_idx = 0
    community_idx = 0

    while len(hidden_edges) < num_hidden_edges:
        if edge_type_idx >= len(edge_types):
            break  # No more edge types available

        local_edge = edge_types[edge_type_idx]
        # Convert to global node indices
        global_edge = (community_idx * 6 + local_edge[0],
                       community_idx * 6 + local_edge[1])
        hidden_edges.append(global_edge)

        community_idx += 1
        if community_idx >= num_communities:
            community_idx = 0
            edge_type_idx += 1

    print(f"[learn_multiple_edges] Hiding {len(hidden_edges)} edges (spread across communities)")
    print(f"[learn_multiple_edges] Hidden edges: {hidden_edges}")
    print(f"[learn_multiple_edges] Total nodes (with {dummy_nodes} dummy): {total_nodes}")

    # Create init_logits: hidden edge positions start at neutral
    init_logits = torch.zeros(total_nodes, total_nodes)
    for (i, j) in hidden_edges:
        init_logits[i, j] = 0.0
        init_logits[j, i] = 0.0

    # Learnable mask: only hidden edges can be updated during training
    learnable_mask = torch.zeros(total_nodes, total_nodes, dtype=torch.bool)
    for (i, j) in hidden_edges:
        learnable_mask[i, j] = True
        learnable_mask[j, i] = True

    # Fixed adjacency: the true graph values
    fixed_adj = true_adj.clone()

    # Fixed mask: all positions are fixed EXCEPT hidden edges
    # This prevents false positives (non-edge positions fixed to 0)
    fixed_mask = torch.ones(total_nodes, total_nodes, dtype=torch.bool)

    # Only hidden edges are learnable (not fixed)
    for (i, j) in hidden_edges:
        fixed_mask[i, j] = False
        fixed_mask[j, i] = False

    return init_logits, learnable_mask, fixed_adj, fixed_mask, hidden_edges

def create_learn_multiple_edges_hard_prior(dataset, num_hidden_edges, seed=42, dummy_nodes=0):
    """
    Create prior for learning multiple adjacent edges forming a connected cluster.

    Edges are selected in order:
    1. All edges within community 0 (BFS-like, up to 9 edges)
    2. Inter-community edge from community 0 to community 1
    3. All edges within community 1 (BFS-like, up to 9 edges)
    4. Inter-community edge from community 1 to community 2
    5. Continue pattern...

    This ensures all hidden edges form a connected subgraph spanning communities.

    Returns:
        init_logits: Initial logit values for the edge scorer
        learnable_mask: Boolean mask for which logits can be updated
        fixed_adj: The adjacency values to use for fixed (non-sampled) edges
        fixed_mask: Boolean mask for which edges are fixed vs sampled
        hidden_edges: List of (i, j) tuples of hidden edges
    """
    edge_index = dataset.G.edge_index
    n_nodes = dataset.G.num_nodes
    total_nodes = n_nodes + dummy_nodes
    num_communities = n_nodes // 6

    # Create true adjacency matrix (including space for dummy nodes)
    true_adj = torch.zeros(total_nodes, total_nodes)
    for k in range(edge_index.shape[1]):
        src, dst = edge_index[0, k], edge_index[1, k]
        true_adj[src, dst] = 1.0
        true_adj[dst, src] = 1.0

    # Local edges within a community (relative to community start node)
    local_community_edges = [
        (0, 1), (1, 2), (3, 4),  # Slashes
        (1, 3), (2, 4), (4, 5),  # Backslashes
        (0, 3), (1, 4), (3, 5),  # Horizontal
    ]

    hidden_edges = []

    for community_idx in range(num_communities):
        if len(hidden_edges) >= num_hidden_edges:
            break

        community_offset = community_idx * 6

        # Get global edges for this community
        community_edges = [(e[0] + community_offset, e[1] + community_offset)
                          for e in local_community_edges]

        # BFS-like selection within this community
        available_edges = set(community_edges)
        covered_nodes = set()

        # Start with first edge in community
        first_edge = (community_offset + 0, community_offset + 1)
        if len(hidden_edges) < num_hidden_edges:
            hidden_edges.append(first_edge)
            available_edges.remove(first_edge)
            covered_nodes.update(first_edge)

        # Greedily select edges that connect to already covered nodes
        while len(hidden_edges) < num_hidden_edges and available_edges:
            adjacent_edges = [e for e in available_edges
                             if e[0] in covered_nodes or e[1] in covered_nodes]

            if not adjacent_edges:
                break

            next_edge = min(adjacent_edges)
            hidden_edges.append(next_edge)
            available_edges.remove(next_edge)
            covered_nodes.update(next_edge)

        # Add inter-community edge to next community (if not last community and still need more)
        if community_idx < num_communities - 1 and len(hidden_edges) < num_hidden_edges:
            # Inter-community edge: last node of current community to first node of next
            inter_edge = (community_offset + 5, community_offset + 6)
            # Verify this edge exists in the graph
            if true_adj[inter_edge[0], inter_edge[1]] == 1.0:
                hidden_edges.append(inter_edge)

    print(f"[learn_multiple_edges_hard] Hiding {len(hidden_edges)} adjacent edges (BFS across communities)")
    print(f"[learn_multiple_edges_hard] Hidden edges: {hidden_edges}")
    print(f"[learn_multiple_edges_hard] Total nodes (with {dummy_nodes} dummy): {total_nodes}")

    # Create init_logits: hidden edge positions start at neutral
    init_logits = torch.zeros(total_nodes, total_nodes)
    for (i, j) in hidden_edges:
        init_logits[i, j] = 0.0
        init_logits[j, i] = 0.0

    # Learnable mask: only hidden edges can be updated during training
    learnable_mask = torch.zeros(total_nodes, total_nodes, dtype=torch.bool)
    for (i, j) in hidden_edges:
        learnable_mask[i, j] = True
        learnable_mask[j, i] = True

    # Fixed adjacency: the true graph values
    fixed_adj = true_adj.clone()

    # Fixed mask: all positions are fixed EXCEPT hidden edges
    # This prevents false positives (non-edge positions fixed to 0)
    fixed_mask = torch.ones(total_nodes, total_nodes, dtype=torch.bool)

    # Only hidden edges are learnable (not fixed)
    for (i, j) in hidden_edges:
        fixed_mask[i, j] = False
        fixed_mask[j, i] = False

    return init_logits, learnable_mask, fixed_adj, fixed_mask, hidden_edges


def run_experiment(cfg):
    """ Runs the latent graph learning experiment with the given config and the number of times. """

    original_run_dir = cfg.run.dir
    num_runs = cfg.num_runs
    complexity_metrics = {i:{} for i in range(1,11)}

    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Starting run {run_idx+1}/{num_runs}")
        print(f"{'='*60}")
        cfg.run.dir = f"{original_run_dir}/run_{run_idx+1}"

        dataset = get_dataset(cfg)

        if cfg.graph_mode == 'imle':
            gm_class = IMLEGraphModule
        elif cfg.graph_mode == 'aimle':
            gm_class = AIMLEGraphModule
        else:
            gm_class = GraphModule

        ########################################
        # data module                          #
        ########################################

        adj = None

        torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                            connectivity=adj,
                                            mask=dataset.mask.bool(),
                                            horizon=cfg.horizon,
                                            window=cfg.window,
                                            stride=cfg.stride)

        dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers={'target': StandardScaler(axis=(0, 1))},
            splitter=dataset.get_splitter(**cfg.dataset.splits),
            batch_size=cfg.batch_size,
            workers=cfg.workers
        )

        dm.setup()
        ########################################
        # predictor                            #
        ########################################

        gm_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        mode=cfg.graph_mode)

        gm_kwargs.update(cfg.graph_module.hparams)

        # Handle learn_one_edge experiment: add fixed adjacency and learnable mask
        if cfg.experiment_name == 'learn_one_edge':
            hidden_edge_idx = getattr(cfg, 'hidden_edge_idx', 0)
            seed = getattr(cfg, 'hidden_edge_seed', 42)
            dummy_nodes = cfg.graph_module.hparams.get('dummy_nodes', 0)
            init_logits, learnable_mask, fixed_adj, fixed_mask, hidden_edge = \
                create_learn_one_edge_prior(dataset, hidden_edge_idx=hidden_edge_idx, seed=seed, dummy_nodes=dummy_nodes)
            gm_kwargs.update({
                'init_logits': init_logits,
                'learnable_mask': learnable_mask,
                'fixed_adj': fixed_adj,
                'fixed_mask': fixed_mask
            })
            print(f"[learn_one_edge] Graph module configured to learn edge {hidden_edge}")

        # Handle learn_multiple_edges experiment: edges spread across communities
        elif cfg.experiment_name == 'learn_multiple_edges':
            seed = getattr(cfg, 'hidden_edge_seed', 42)
            num_hidden_edges = getattr(cfg, 'num_hidden_edges', 5)
            dummy_nodes = cfg.graph_module.hparams.get('dummy_nodes', 0)
            init_logits, learnable_mask, fixed_adj, fixed_mask, hidden_edges = \
                create_learn_multiple_edges_prior(dataset, num_hidden_edges=num_hidden_edges, seed=seed, dummy_nodes=dummy_nodes)
            gm_kwargs.update({
                'init_logits': init_logits,
                'learnable_mask': learnable_mask,
                'fixed_adj': fixed_adj,
                'fixed_mask': fixed_mask
            })
            print(f"[learn_multiple_edges] Graph module configured to learn edges {hidden_edges}")

        # Handle learn_multiple_edges_hard experiment: adjacent edges clustered together
        elif cfg.experiment_name == 'learn_multiple_edges_hard':
            seed = getattr(cfg, 'hidden_edge_seed', 42)
            num_hidden_edges = getattr(cfg, 'num_hidden_edges', 5)
            dummy_nodes = cfg.graph_module.hparams.get('dummy_nodes', 0)
            init_logits, learnable_mask, fixed_adj, fixed_mask, hidden_edges = \
                create_learn_multiple_edges_hard_prior(dataset, num_hidden_edges=num_hidden_edges, seed=seed, dummy_nodes=dummy_nodes)
            gm_kwargs.update({
                'init_logits': init_logits,
                'learnable_mask': learnable_mask,
                'fixed_adj': fixed_adj,
                'fixed_mask': fixed_mask
            })
            print(f"[learn_multiple_edges_hard] Graph module configured to learn edges {hidden_edges}")

        loss_fn = MaskedMAE()

        metrics = {'mae': MaskedMAE(),
                'mse': MaskedMSE(),
                'mape': MaskedMAPE()}
        
        eval_mode = 'sampling'

        if cfg.graph_mode == 'sf':
            if cfg.use_baseline == 'frechet':
                eval_mode = 'mode'
            else:
                eval_mode = 'sampling'

        if cfg.graph_mode == 'pd' or cfg.graph_mode == 'st':
            predictor_class = LatentGraphPredictor
            pred_kwargs = dict(graph_module_class=gm_class,
                            graph_module_kwargs=gm_kwargs,
                            mc_samples=cfg.mc_samples,
                            )

        elif cfg.graph_mode == 'sf':
            predictor_class = SFGraphPredictor
            pred_kwargs = dict(sf_weight=cfg.sf_weight,
                            graph_module_class=gm_class,
                            graph_module_kwargs=gm_kwargs,
                            use_baseline=cfg.use_baseline,
                            mc_samples=cfg.mc_samples,
                            eval_mode=eval_mode,
                            variance_reduced=cfg.variance_reduced,
                            surrogate_lam=cfg.lam,
                            )
            if cfg.use_baseline == 'doublecv':
                doublecv_args_dict = {'gradient_level':cfg.gradient_level, 'doublecv_mode':cfg.doublecv_mode, 'use_taylor':cfg.use_taylor}
                pred_kwargs.update(doublecv_args_dict)

        elif cfg.graph_mode == 'imle':
            predictor_class = IMLEGraphPredictor
            pred_kwargs = dict(graph_module_class=gm_class,
                            graph_module_kwargs=gm_kwargs,
                            mc_samples=cfg.mc_samples,
                            eval_mode=eval_mode,
                            clip_grad=cfg.clip_grad,
                            clip_grad_val=0.5)

        elif cfg.graph_mode == 'aimle':
            predictor_class = AIMLEGraphPredictor
            pred_kwargs = dict(graph_module_class=gm_class,
                            graph_module_kwargs=gm_kwargs,
                            mc_samples=cfg.mc_samples,
                            eval_mode=eval_mode,
                            clip_grad=cfg.clip_grad,
                            clip_grad_val=0.5)


        else:
            raise NotImplementedError(f"Graph learning mode {cfg.graph_mode} not available.")

        model_cls = dataset.model_class
        if cfg.experiment_name == 'graph_id':
            model_kwargs = dataset.model_kwargs
        elif cfg.experiment_name == 'learn_one_edge':
            # Same as graph_id: use true filter coefficients, only learn one edge
            model_kwargs = dataset.model_kwargs
        elif cfg.experiment_name == 'joint':
            model_kwargs = dict(
                spatial_order=3,
                temporal_order=4
            )
        elif cfg.experiment_name == 'joint_medium_1':
            # Correct spatial order, more temporal overparameterization than joint
            model_kwargs = dict(
                spatial_order=3,
                temporal_order=6
            )
        elif cfg.experiment_name == 'joint_medium_2':
            # Overparameterized spatial order, same temporal as joint
            model_kwargs = dict(
                spatial_order=4,
                temporal_order=4
            )
        elif cfg.experiment_name == 'joint_hard':
            model_kwargs = dict(
                spatial_order=4,
                temporal_order=6
            )
        elif cfg.experiment_name == 'joint_noisy':
            # Start from true coefficients with Gaussian noise; coefficients are learnable
            filter_coef_noise_std = getattr(cfg, 'filter_coef_noise_std', 1.0)
            true_coefs = torch.tensor([[5, 2], [-4, 6], [-1, 0]], dtype=torch.float32) # Same coefficients as graph_id.
            noisy_coefs = true_coefs + torch.randn_like(true_coefs) * filter_coef_noise_std # Perturbing the ground truth coefs with gaussian noise.
            model_kwargs = dict(
                filter_coefs=noisy_coefs,
                filter_coefs_learnable=True
            )
        elif cfg.experiment_name == 'learn_multiple_edges':
            # Same as graph_id: use true filter coefficients, only learn specified edges
            model_kwargs = dataset.model_kwargs
        elif cfg.experiment_name == 'learn_multiple_edges_hard':
            # Same as graph_id: use true filter coefficients, only learn specified edges
            model_kwargs = dataset.model_kwargs
        elif cfg.experiment_name == 'joint_stgnn':
            model_cls = TTSModel
            model_kwargs = dict(
                input_size=torch_dataset.n_channels,
                hidden_size=32,
                output_size=1,
                horizon=cfg.horizon,
                n_nodes=torch_dataset.n_nodes,
                enc_layers=1,
                gcn_layers=2,
            )
        else:
            raise NotImplementedError(f"Experiment {cfg.experiment_name} not avaiable.")

        scheduler_class = None
        scheduler_kwargs = None

        predictor = predictor_class(
            model_class=model_cls,
            model_kwargs=model_kwargs,
            optim_class=torch.optim.Adam,
            optim_kwargs=dict(cfg.optimizer.hparams),
            loss_fn=loss_fn,
            metrics=metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scale_target=False,
            **pred_kwargs
        )

        print('eval_mode for test is:', predictor.eval_mode)

        ########################################
        # training                             #
        ########################################

        early_stop_callback = EarlyStopping(
            monitor='val_mae',
            patience=cfg.patience,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.run.dir,
            save_top_k=1,
            monitor='val_mae',
            mode='min',
        )

        lr_monitor = LearningRateMonitor(
            logging_interval='epoch'
        )

        batches_epoch = 1.0 if cfg.batches_epoch < 0 else cfg.batches_epoch
        trainer = pl.Trainer(max_epochs=cfg.epochs,
                            limit_train_batches=batches_epoch,
                            default_root_dir=cfg.run.dir,
                            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                            callbacks=[early_stop_callback,
                                        checkpoint_callback,
                                        lr_monitor],
                            gradient_clip_algorithm='value',
                            gradient_clip_val=.5 if (cfg.clip_grad and cfg.graph_mode != 'aimle') else None)

        tsl.logger.info(f"Optimal MAE (analytical) {dataset.mae_optimal}")

        print("Checking model parameters for requires_grad:")
        for name, param in predictor.named_parameters():
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

        ### TIME AND MEMORY MEASUREMENTS MUST START HERE ###
        start_time = time.time()
        #initial_gpu_memory = get_gpu_memory_usage()
        #initial_cpu_memory = get_memory_usage()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        trainer.fit(predictor,
                    train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())

        actual_epochs = trainer.current_epoch + 1  # Adding 1 since epochs are 0-indexed

        training_end_time = time.time()
        training_total_time = training_end_time - start_time
        #training_end_gpu_memory = get_gpu_memory_usage()
        #training_memory_use = training_end_gpu_memory - initial_gpu_memory
        ########################################
        # testing                              #
        ########################################

        #start_test_time = time.time()
        #start_test_gpu_memory = get_gpu_memory_usage()

        #predictor.load_state_dict(
        #    torch.load(checkpoint_callback.best_model_path,
        #               lambda storage, loc: storage)['state_dict'])

        #predictor.freeze()
        #trainer.test(predictor, dataloaders=dm.test_dataloader())
        trainer.test(ckpt_path=checkpoint_callback.best_model_path, dataloaders=dm.test_dataloader())

        ### TIME AND MEMORY MEASUREMENTS MUST END HERE ###
        #test_end_time = time.time()
        #test_total_time = test_end_time - start_test_time
        #test_end_gpu_memory = get_gpu_memory_usage()
        #test_memory_use = test_end_gpu_memory - start_test_gpu_memory

        final_time = time.time()
        total_time = final_time - start_time
        #final_gpu_memory = get_gpu_memory_usage()
        #total_gpu_memory_use = final_gpu_memory - initial_gpu_memory
        peak_memory_usage = torch.cuda.max_memory_allocated() / (1024**2)

        complexity_metrics[run_idx+1]['training_time'] = training_total_time
        #complexity_metrics[run_idx+1]['training_memory'] = training_memory_use
        #complexity_metrics[run_idx+1]['test_time'] = test_total_time
        #complexity_metrics[run_idx+1]['test_memory'] = test_memory_use
        complexity_metrics[run_idx+1]['total_time'] = total_time
        complexity_metrics[run_idx+1]['total_memory'] = peak_memory_usage
        complexity_metrics[run_idx+1]['actual_epochs'] = actual_epochs
        complexity_metrics[run_idx+1]['time_per_epoch'] = complexity_metrics[run_idx+1]['training_time'] / complexity_metrics[run_idx+1]['actual_epochs']

        ########################################
        # visualization                        #
        ########################################

        if not getattr(cfg, 'visualize', False):
            print("[Visualization] Skipped (visualize=False)")
        else:
            try:
                # Load best checkpoint for visualization
                best_ckpt = torch.load(checkpoint_callback.best_model_path, map_location='cpu', weights_only=False)
                predictor.load_state_dict(best_ckpt['state_dict'])

                # Run forward pass to get sampled graph for visualization
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                predictor = predictor.to(device)
                predictor.eval()
                predictor.freeze()

                # Temporarily set eval_mode to 'sampling' to ensure test_graphs is populated
                original_eval_mode = predictor.eval_mode
                predictor.eval_mode = 'sampling'

                with torch.no_grad():
                    test_loader = dm.test_dataloader()
                    batch = next(iter(test_loader)).to(device)
                    _ = predictor.predict_batch(batch, preprocess=False, postprocess=False)

                # Restore original eval_mode
                predictor.eval_mode = original_eval_mode

                # Get ground truth graph from dataset
                gt_edge_index = dataset.G.edge_index
                n_nodes = dataset.G.num_nodes
                node_positions = dataset.G.node_position
                gt_adj = edge_index_to_adjacency(gt_edge_index, n_nodes)

                # Get the learned adjacency from the first MC sample (stored during testing)
                if predictor.test_graphs is None or len(predictor.test_graphs) == 0:
                    raise ValueError("No test graphs stored. Run predictor forward pass to populate test_graphs.")

                # Get first MC sample and convert to numpy, slicing to n_nodes to exclude dummy nodes
                test_adj = predictor.test_graphs[0]
                if test_adj.dim() == 3:
                    test_adj = test_adj[0]  # Take first batch element if batched
                learned_adj = test_adj.numpy()
                # Slice to n_nodes if needed (to exclude dummy nodes)
                if learned_adj.shape[0] > n_nodes:
                    learned_adj = learned_adj[:n_nodes, :n_nodes]

                # Handle initial_adj and hidden_edge for learn_one_edge and learn_multiple_edges experiments
                initial_adj = None
                hidden_edge = None
                if cfg.experiment_name == 'learn_one_edge':
                    # Initial graph is ground truth with hidden edge removed
                    # Use dummy_nodes=0 for visualization (we only visualize real nodes)
                    hidden_edge_idx = getattr(cfg, 'hidden_edge_idx', 0)
                    seed = getattr(cfg, 'hidden_edge_seed', 42)
                    _, _, fixed_adj, _, hidden_edge = create_learn_one_edge_prior(
                        dataset, hidden_edge_idx=hidden_edge_idx, seed=seed, dummy_nodes=0
                    )
                    initial_adj = fixed_adj.numpy()
                elif cfg.experiment_name == 'learn_multiple_edges':
                    seed = getattr(cfg, 'hidden_edge_seed', 42)
                    num_hidden_edges = getattr(cfg, 'num_hidden_edges', 5)
                    _, _, fixed_adj, _, hidden_edge = create_learn_multiple_edges_prior(
                        dataset, num_hidden_edges=num_hidden_edges, seed=seed, dummy_nodes=0
                    )
                    initial_adj = fixed_adj.numpy()
                elif cfg.experiment_name == 'learn_multiple_edges_hard':
                    seed = getattr(cfg, 'hidden_edge_seed', 42)
                    num_hidden_edges = getattr(cfg, 'num_hidden_edges', 5)
                    _, _, fixed_adj, _, hidden_edge = create_learn_multiple_edges_hard_prior(
                        dataset, num_hidden_edges=num_hidden_edges, seed=seed, dummy_nodes=0
                    )
                    initial_adj = fixed_adj.numpy()

                # Create visualizations
                save_dir = cfg.run.dir
                visualize_graphs(
                    ground_truth_adj=gt_adj,
                    learned_adj=learned_adj,
                    node_positions=node_positions,
                    save_path=f"{save_dir}/graph_comparison.png",
                    initial_adj=initial_adj,
                    hidden_edge=hidden_edge
                )
                visualize_edge_comparison(
                    ground_truth_adj=gt_adj,
                    learned_adj=learned_adj,
                    node_positions=node_positions,
                    save_path=f"{save_dir}/edge_comparison.png",
                    hidden_edge=hidden_edge
                )
                print(f"[Visualization] Graphs saved to {save_dir}/")

            except Exception as e:
                print(f"[Visualization] Warning: Could not generate visualizations: {e}")

    summary_file = f"{original_run_dir}/complexity_summary.json"
    with open(summary_file, 'w') as f:
            json.dump(complexity_metrics, f, indent=2, default=str)

if __name__ == '__main__':
    exp = Experiment(run_fn=run_experiment, config_path='config/synthetic')
    exp.run()