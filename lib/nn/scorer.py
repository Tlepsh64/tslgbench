import torch
from torch import nn
from lib.utils.utils import soft_clip
from einops import repeat

class AdjEmb(nn.Module):
    """
    Learnable adjacency matrix embedding.

    Args:
        num_nodes: Number of nodes in the graph
        learnable: Whether logits are learnable (default True)
        clamp_at: Value to soft-clip logits at (default 5.0)
        init_logits: Optional initial logits tensor (num_nodes, num_nodes).
                     If None, random initialization in [-0.5, 0.5]
        learnable_mask: Optional boolean mask (num_nodes, num_nodes).
                       True = learnable, False = frozen. If None, all learnable.
    """

    def __init__(self,
                 num_nodes,
                 learnable=True,
                 clamp_at=5.,
                 init_logits=None,
                 learnable_mask=None):
        super(AdjEmb, self).__init__()
        self.clamp_value = clamp_at
        self.num_nodes = num_nodes

        # Initialize logits
        if init_logits is not None:
            assert init_logits.shape == (num_nodes, num_nodes), \
                f"init_logits shape {init_logits.shape} doesn't match ({num_nodes}, {num_nodes})"
            initial_values = init_logits.clone()
        else:
            initial_values = torch.rand(num_nodes, num_nodes) - 0.5

        # Handle learnable mask
        if learnable_mask is not None:
            assert learnable_mask.shape == (num_nodes, num_nodes), \
                f"learnable_mask shape {learnable_mask.shape} doesn't match ({num_nodes}, {num_nodes})"
            self.register_buffer('learnable_mask', learnable_mask.bool())
            self.register_buffer('frozen_logits', initial_values.clone())
            # Only the learnable positions need gradients
            self.logits = nn.Parameter(initial_values, requires_grad=learnable)
        else:
            self.learnable_mask = None
            self.frozen_logits = None
            self.logits = nn.Parameter(initial_values, requires_grad=learnable)

    def forward(self, x, *args, **kwargs):
        """"""
        b, *_ = x.size()

        # If we have a learnable mask, combine frozen and learnable logits
        if self.learnable_mask is not None:
            # Use frozen values where mask is False, learnable where mask is True
            combined_logits = torch.where(self.learnable_mask, self.logits, self.frozen_logits)
            scores = soft_clip(combined_logits, self.clamp_value)
        else:
            scores = soft_clip(self.logits, self.clamp_value)

        return repeat(scores, '... -> b ...', b=b)