import torch
from torch import nn

from lib.nn.graph_samplers.bes import ConcreteBinarySampler, \
                                      StraightThroughBinarySampler, \
                                      BinaryEdgeSampler
from lib.nn.graph_samplers.imle import IMLESampler, AIMLESampler
from lib.nn.graph_samplers.sns import SubsetNeighborhoodSampler, StraightThroughSubsetSampler
from lib.utils.utils import adjs_to_edge_index, adjs_to_fc_edge_index


class GraphSampler(nn.Module):
    def __init__(self, mode, sampler_type, k=None, tau=1., dummy_nodes=0, return_fc_edge_index = False):
        super(GraphSampler, self).__init__()
        if mode == 'pd':
            if sampler_type == 'bes':
                self.k = None
                self.sampler = ConcreteBinarySampler()
            else:
                raise NotImplementedError('Sampler {} not implemented for mode {}'.format(sampler_type, mode))
        elif mode == 'st':
            if sampler_type == 'bes':
                self.k = None
                self.sampler = StraightThroughBinarySampler()
            elif sampler_type == 'sns':
                self.k = k
                self.sampler = StraightThroughSubsetSampler(k=k)
            else:
                raise NotImplementedError('Sampler {} not implemented for mode {}'.format(sampler_type, mode))
        elif mode == 'sf':
            if sampler_type == 'sns':
                self.k = k
                self.sampler = SubsetNeighborhoodSampler(k=k)
            elif sampler_type == 'bes':
                self.k = None
                self.sampler = BinaryEdgeSampler()
            else:
                raise NotImplementedError('Sampler {} not implemented for mode {}'.format(sampler_type, mode))
        tau = torch.tensor(tau)
        self.register_buffer('tau', tau)
        self.dummy_nodes = dummy_nodes
        self.sampling_mode = mode
        self.return_fc_edge_index = return_fc_edge_index

    def forward(self, scores):
        adj, ll = self.sampler(scores, tau=self.tau)    
        adj, edge_index = self.to_connectivity(adj)
        if self.dummy_nodes > 0 and ll is not None:
            assert ll.shape[-1] == adj.shape[-1] + self.dummy_nodes
            ll = ll[..., :-self.dummy_nodes]
        return adj, edge_index, ll

    def to_connectivity(self, adjs):
        # remove dummy nodes
        if self.dummy_nodes > 0:
            adjs = adjs[..., :-self.dummy_nodes, :-self.dummy_nodes]

        # Add self-loops to prevent zero-degree nodes. A zero-degree node makes the
        # GCN degree-normalization divide by zero: the forward value is guarded
        # (inf -> 0), but the BACKWARD derivative (~deg^-3/2) is not, producing NaN
        # gradients that corrupt the logits. Adding I guarantees deg >= 1 everywhere,
        # so both forward and backward stay finite.
        #n_nodes = adjs.shape[-1]
        #eye = torch.eye(n_nodes, device=adjs.device, dtype=adjs.dtype)
        #if adjs.dim() == 3:
        #    eye = eye.unsqueeze(0).expand(adjs.size(0), -1, -1)
        #adjs = adjs + eye

        #fc_edge_index, fc_edge_weight = None, None
        if self.sampling_mode == 'sf':
            edge_index, edge_weight = adjs_to_edge_index(adjs)
            if self.return_fc_edge_index:
                fc_edge_index, fc_edge_weight = adjs_to_fc_edge_index(adjs)
        # ST+SNS sparse optimization: use sparse edge_index since adj is k-hot.
        # To revert to original (dense N² edges), comment this elif block
        # and uncomment the "Original ST dense" else block below.
        #elif self.sampling_mode == 'st': # Seems st+bes suffers from gradient instability too. Adjust this.
        #    edge_index, edge_weight = adjs_to_edge_index(adjs)
        else:
            edge_index, edge_weight = adjs_to_fc_edge_index(adjs)
        # --- Original ST dense (uncomment to revert, comment the elif above) ---
        #else:
        #    edge_index, edge_weight = adjs_to_fc_edge_index(adjs)

        return adjs, (edge_index, edge_weight)

    def mode(self, scores):
        adj = self.sampler.mode(scores, tau=self.tau)
        return self.to_connectivity(adj)

class IMLEGraphSampler(nn.Module):
    def __init__(self, nb_samples, mode, lambda_val=10.0, noise_temp=1.0, tau=1.0, dummy_nodes=0):
        super().__init__()
        self.lambda_val = lambda_val
        self.noise_temp = torch.tensor(noise_temp)
        self.tau = torch.tensor(tau)
        self.dummy_nodes = dummy_nodes
        #self.mode = mode  # Our custom mode
        self.nb_samples = nb_samples
        
        # Use our IMLE sampler
        self.sampler = IMLESampler(lambda_val=self.lambda_val, nb_samples=self.nb_samples)

    def forward(self, scores):
        #
        #scores: [batch_size, n_nodes, n_nodes] 
        #Returns: adj, edge_index, ll (log-likelihood)
        
        # Sample using IMLE
        adj, ll = self.sampler(scores, noise_temp=self.noise_temp)
        
        # Convert to connectivity format using Cini's utilities
        adj, edge_index = self.to_connectivity(adj)
        
        # Remove dummy nodes from log-likelihood if needed
        if self.dummy_nodes > 0 and ll is not None:
            assert ll.shape[-1] == adj.shape[-1] + self.dummy_nodes
            ll = ll[..., :-self.dummy_nodes]
            
        return adj, edge_index, ll

    def to_connectivity(self, adjs):
        #Convert dense adjacency to sparse format using Cini's utilities
        # Remove dummy nodes
        if self.dummy_nodes > 0:
            adjs = adjs[..., :-self.dummy_nodes, :-self.dummy_nodes]

        # Use Cini's utility functions
        from lib.utils.utils import adjs_to_edge_index
        edge_index, edge_weight = adjs_to_edge_index(adjs)

        return adjs, (edge_index, edge_weight)

    def mode(self, scores):
        #For evaluation, use the mode
        adj = self.sampler.mode(scores, tau=self.tau)
        return self.to_connectivity(adj)

class AIMLEGraphSampler(nn.Module):
    def __init__(self, nb_samples, mode, noise_temp=1.0, tau=1.0, dummy_nodes=0,
                 nb_marginal_samples=1, symmetric_perturbation=False,
                 map_solver="topk", use_momentum=False, k=5):
        """
        AIMLE Graph Sampler for latent graph learning.

        Args:
            nb_samples: Number of independent noise samples for gradient estimation
            mode: Sampling mode (unused, kept for compatibility)
            noise_temp: Temperature for noise perturbation
            tau: Temperature for mode computation
            dummy_nodes: Number of dummy nodes to remove
            nb_marginal_samples: Number of samples to average for marginal estimation
            symmetric_perturbation: If True, use central difference; if False, use forward difference
            map_solver: MAP solver to use - "topk" for k-subset selection, "kruskal" for MST
            use_momentum: If True, use momentum for adaptive β updates
            k: Number of edges per node for topk solver
        """
        super().__init__()
        self.noise_temp = torch.tensor(noise_temp)
        self.tau = torch.tensor(tau)
        self.dummy_nodes = dummy_nodes
        self.nb_samples = nb_samples
        self.nb_marginal_samples = nb_marginal_samples
        self.symmetric_perturbation = symmetric_perturbation
        self.map_solver = map_solver
        self.use_momentum = use_momentum

        # Use our AIMLE sampler with all parameters
        self.sampler = AIMLESampler(
            noise_temp=noise_temp,
            nb_samples=self.nb_samples,
            nb_marginal_samples=self.nb_marginal_samples,
            symmetric_perturbation=self.symmetric_perturbation,
            map_solver=self.map_solver,
            use_momentum=self.use_momentum,
            k=k
        )

    def forward(self, scores):
        #
        #scores: [batch_size, n_nodes, n_nodes] 
        #Returns: adj, edge_index, ll (log-likelihood)
        
        # Sample using IMLE
        adj, ll = self.sampler(scores, noise_temp=self.noise_temp)
        
        # Convert to connectivity format using Cini's utilities
        adj, edge_index = self.to_connectivity(adj)
        
        # Remove dummy nodes from log-likelihood if needed
        if self.dummy_nodes > 0 and ll is not None:
            assert ll.shape[-1] == adj.shape[-1] + self.dummy_nodes
            ll = ll[..., :-self.dummy_nodes]
            
        return adj, edge_index, ll

    def to_connectivity(self, adjs):
        #Convert dense adjacency to sparse format using Cini's utilities
        # Remove dummy nodes
        if self.dummy_nodes > 0:
            adjs = adjs[..., :-self.dummy_nodes, :-self.dummy_nodes]

        # Use Cini's utility functions
        from lib.utils.utils import adjs_to_edge_index
        edge_index, edge_weight = adjs_to_edge_index(adjs)

        return adjs, (edge_index, edge_weight)

    def mode(self, scores):
        #For evaluation, use the mode
        adj = self.sampler.mode(scores, tau=self.tau)
        return self.to_connectivity(adj)
    