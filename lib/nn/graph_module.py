import torch
from torch.distributions import Bernoulli
from lib.nn.graph_samplers.graph_sampler import GraphSampler, IMLEGraphSampler, AIMLEGraphSampler
from lib.nn.scorer import AdjEmb
from tsl.ops.connectivity import adj_to_edge_index
from tsl.nn.models import BaseModel


class GraphModule(BaseModel):
    def __init__(self,
                 n_nodes,
                 sampler,
                 mode,
                 k=10,
                 dummy_nodes=0,
                 tau=1.,
                 init_logits=None,
                 learnable_mask=None,
                 fixed_adj=None,
                 fixed_mask=None):
        """
        Args:
            fixed_adj: (n_nodes, n_nodes) tensor of known adjacency values
            fixed_mask: (n_nodes, n_nodes) bool tensor. True = use fixed_adj value (no sampling)
        """
        super(GraphModule, self).__init__()

        self.mode = mode
        self.edge_scorer = AdjEmb(num_nodes=n_nodes + dummy_nodes,
                                  init_logits=init_logits,
                                  learnable_mask=learnable_mask)

        self.sampler = GraphSampler(mode=mode,
                                    sampler_type=sampler,
                                    tau=tau,
                                    k=k,
                                    dummy_nodes=dummy_nodes)

        self.dummy_nodes = dummy_nodes

        # Register fixed adjacency buffers for learn_one_edge experiment
        if fixed_adj is not None and fixed_mask is not None:
            self.register_buffer('fixed_adj', fixed_adj)
            self.register_buffer('fixed_mask', fixed_mask)
        else:
            self.fixed_adj = None
            self.fixed_mask = None

    def forward(self, x, **kwargs):

        logits = self.edge_scorer(x)

        #adj, (edge_index, edge_weight), (fc_edge_index, fc_edge_weight), ll = self.sampler(logits)
        adj, (edge_index, edge_weight), ll = self.sampler(logits)

        # Apply fixed adjacency override and recompute ll (for learn_one_edge experiment)
        if self.fixed_adj is not None and self.fixed_mask is not None:
            # adj shape: (batch, n_nodes, n_nodes) without dummy nodes
            # fixed_adj/fixed_mask shape: (total_nodes, total_nodes) with dummy nodes
            # Trim fixed_adj/fixed_mask to match adj (remove dummy node portion)
            n_nodes = adj.shape[-1]
            fixed_adj_trimmed = self.fixed_adj[:n_nodes, :n_nodes]
            fixed_mask_trimmed = self.fixed_mask[:n_nodes, :n_nodes]

            # Recompute ll with masking (only include unfixed edges)
            if ll is not None and self.training:
                tau = self.sampler.tau
                # ll_mask: True for edges that should be sampled (unfixed edges)
                ll_mask = ~fixed_mask_trimmed  # (n_nodes, n_nodes)

                # Compute log_prob using trimmed logits and already-sampled adj
                logits_trimmed = logits[:, :n_nodes, :n_nodes] if self.dummy_nodes > 0 else logits
                dist = Bernoulli(logits=logits_trimmed / tau)
                log_probs = dist.log_prob(adj)  # (batch, n_nodes, n_nodes)

                # Apply mask: zero out log_probs for fixed edges
                masked_log_probs = log_probs * ll_mask.unsqueeze(0).float()

                # Sum over last dimension (like original BES)
                ll = masked_log_probs.sum(-1)  # (batch, n_nodes)

            # Override sampled values with fixed values where mask is True
            if adj.dim() == 3:
                fixed_adj_expanded = fixed_adj_trimmed.unsqueeze(0).expand_as(adj)
                fixed_mask_expanded = fixed_mask_trimmed.unsqueeze(0).expand_as(adj)
            else:
                fixed_adj_expanded = fixed_adj_trimmed
                fixed_mask_expanded = fixed_mask_trimmed
            adj = torch.where(fixed_mask_expanded, fixed_adj_expanded, adj)
            # Recompute edge_index and edge_weight from corrected adjacency
            edge_index, edge_weight = adj_to_edge_index(adj)

        _, (mean_edge_index, mean_edge_weight) = self.sampler.mode(logits)
        mean_graph = dict(edge_index=mean_edge_index,
                          edge_weight=mean_edge_weight)

        return dict(edge_index=edge_index,
                    edge_weight=edge_weight,
                    disjoint=adj.dim() > 2,
                    adj=adj,
                    ll=ll,
                    mean_graph=mean_graph,
                    logits=logits)

class IMLEGraphModule(BaseModel):
    def __init__(self,
                 n_nodes,
                 sampler,
                 mode,
                 lambda_val=10.0,
                 noise_temp=1.0,
                 tau=1.0,
                 nb_samples=1,
                 dummy_nodes=0):
        super().__init__()

        self.mode = mode
        self.edge_scorer = AdjEmb(num_nodes=n_nodes + dummy_nodes)
        self.dummy_nodes = dummy_nodes

        # Create IMLE sampler
        self.sampler = IMLEGraphSampler(lambda_val=lambda_val,
                                        noise_temp=noise_temp,
                                        tau=tau,
                                        nb_samples=nb_samples,
                                        dummy_nodes=dummy_nodes,
                                        mode=mode)

    def forward(self, x, **kwargs):
        # Get logits from edge scorer - these are the parameters we want to learn
        logits = self.edge_scorer(x)  # shape: [batch_size, n_nodes, n_nodes]

        # Sample using IMLE - this creates the autograd node
        adj, (edge_index, edge_weight), ll = self.sampler(logits, **kwargs)

        # Get mean graph for evaluation
        _, (mean_edge_index, mean_edge_weight) = self.sampler.mode(logits)
        mean_graph = dict(edge_index=mean_edge_index,
                          edge_weight=mean_edge_weight)

        return dict(edge_index=edge_index,
                    edge_weight=edge_weight,
                    disjoint=adj.dim() > 2,
                    adj=adj,
                    ll=ll,  # None for IMLE
                    mean_graph=mean_graph,
                    logits=logits)

    def _dense_to_sparse(self, dense_adj):
        #Convert dense adjacency to sparse format - adapt from Cini's code
        # This should match Cini's existing implementation
        # For simplicity, I'll show a basic version
        batch_size, n_nodes, _ = dense_adj.shape
        if batch_size == 1:
            edge_index = dense_adj[0].nonzero(as_tuple=False).t()
            edge_weight = dense_adj[0][dense_adj[0] != 0]
        else:
            # Handle batched case - depends on Cini's specific format
            edge_index = []
            edge_weight = []
            for b in range(batch_size):
                ei = dense_adj[b].nonzero(as_tuple=False).t()
                ew = dense_adj[b][dense_adj[b] != 0]
                edge_index.append(ei)
                edge_weight.append(ew)
        return edge_index, edge_weight
    
class AIMLEGraphModule(BaseModel):
    def __init__(self,
                 n_nodes,
                 sampler,
                 mode,
                 noise_temp=1.0,
                 tau=1.0,
                 nb_samples=1,
                 nb_marginal_samples=1,
                 symmetric_perturbation=False,
                 map_solver="topk",
                 use_momentum=False,
                 dummy_nodes=0,
                 k=5):
        """
        AIMLE Graph Module for latent graph learning.

        Args:
            n_nodes: Number of nodes in the graph
            sampler: Sampler type (unused, kept for compatibility)
            mode: Sampling mode (unused, kept for compatibility)
            noise_temp: Temperature for noise perturbation
            tau: Temperature for mode computation
            nb_samples: Number of independent noise samples for gradient estimation
            nb_marginal_samples: Number of samples to average for marginal estimation
            symmetric_perturbation: If True, use central difference; if False, use forward difference
            map_solver: MAP solver to use - "topk" for k-subset selection, "kruskal" for MST
            use_momentum: If True, use momentum for adaptive β updates
            dummy_nodes: Number of dummy nodes
            k: Number of edges per node for topk solver
        """
        super().__init__()

        self.mode = mode
        self.edge_scorer = AdjEmb(num_nodes=n_nodes + dummy_nodes)
        self.dummy_nodes = dummy_nodes

        # Create AIMLE sampler with all parameters
        self.sampler = AIMLEGraphSampler(
            noise_temp=noise_temp,
            tau=tau,
            nb_samples=nb_samples,
            nb_marginal_samples=nb_marginal_samples,
            symmetric_perturbation=symmetric_perturbation,
            map_solver=map_solver,
            use_momentum=use_momentum,
            dummy_nodes=dummy_nodes,
            mode=mode,
            k=k
        )

    def forward(self, x, **kwargs):
        # Get logits from edge scorer - these are the parameters we want to learn
        logits = self.edge_scorer(x)  # shape: [batch_size, n_nodes, n_nodes]

        # Sample using AIMLE - this creates the autograd node
        adj, (edge_index, edge_weight), ll = self.sampler(logits, **kwargs)

        # Get mean graph for evaluation
        _, (mean_edge_index, mean_edge_weight) = self.sampler.mode(logits)
        mean_graph = dict(edge_index=mean_edge_index,
                          edge_weight=mean_edge_weight)

        return dict(edge_index=edge_index,
                    edge_weight=edge_weight,
                    disjoint=adj.dim() > 2,
                    adj=adj,
                    ll=ll,  # None for AIMLE
                    mean_graph=mean_graph,
                    logits=logits)
    