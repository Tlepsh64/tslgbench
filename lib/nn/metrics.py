from torchmetrics.utilities.checks import _check_same_shape
from functools import partial
import warnings
from torchmetrics import Metric
import torch
from einops import rearrange


class MaskedScoreFuctionLoss(Metric):
    r"""
    Base class to implement the metrics used in `tsl`.

    In particular a `MaskedMetric` accounts for missing values in the input sequences by accepting a boolean mask as
    additional input.

    Args:
        cost_fn: Base function to compute the metric point wise.
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite values.
        compute_on_step (bool, optional): Whether to compute the metric right-away or to accumulate the results.
                         This should be `True` when using the metric to compute a loss function, `False` if the metric
                         is used for logging the aggregate value across different mini-batches.
        at (int, optional): Whether to compute the metric only w.r.t. a certain time step.
    """
    is_differentiable = True
    full_state_update = False
    def __init__(self,
                 cost_fn,
                 variance_reduced=True,
                 cost_momentum=0.1,
                 cost_kwargs=None,
                 lam=None,
                 **kwargs):
        super(MaskedScoreFuctionLoss, self).__init__(**kwargs)

        if cost_kwargs is None:
            cost_kwargs = dict()
        self.cost_fn = partial(cost_fn, **cost_kwargs)
        self.ma_momentum = cost_momentum
        self.variance_reduced = variance_reduced
        self.lam = lam
        self.add_state('value', dist_reduce_fx='sum', default=torch.tensor(0., dtype=torch.float))
        self.add_state('numel', dist_reduce_fx='sum', default=torch.tensor(0., dtype=torch.float))

    def _check_mask(self, mask, val):
        if mask is None:
            mask = torch.ones_like(val, dtype=torch.bool)
        else:
            mask = mask.bool()
            _check_same_shape(mask, val)
        return mask

    def compute_cost(self, y_hat, y, mask):
        _check_same_shape(y_hat, y)
        cost = self.cost_fn(y_hat, y)
        mask = self._check_mask(mask, cost)
        cost = torch.where(mask, cost, torch.zeros_like(cost))
        return cost, mask
    
    def update(self, score, y_hat, y, baseline=None, mask=None):
        # make sure the score has a proper shape
        if score.dim() == 1:
            score = score[None]
        #print('(METRICS)Shape of score before rearranging:', score.shape) # 128,30
        score = rearrange(score, 'b n -> b 1 n 1')
        #print('(METRICS)Shape of score after rearranging:', score.shape) # 128,1,30,1
        # account for
        score_mask = torch.isfinite(score)
        if not score_mask.all():
            warnings.warn("Nan values in scores.")
            score = torch.where(score_mask, score, torch.zeros_like(score))

        # make sure the mask has a proper shape
        mask = self._check_mask(mask, y)
        mask = mask.float() * score_mask.float()

        #print('(METRICS)Predictions shape', y_hat.shape) # 128,1,30,1
        #print('(METRICS)Target shape', y.shape) # 128,1,30,1
        cost, mask = self.compute_cost(y_hat, y, mask)

        if baseline is not None:
            #print('(METRICS)Baseline shape', baseline.shape) # 128,1,30,1 > Cost with the baseline graph.
            #print('(METRICS)Cost shape', cost.shape) # 128,1,30,1 > Cost with the learned/sampled graph.
            _check_same_shape(baseline, cost)
            cost = cost - baseline

        # OLD: compute baseline
        #if y_b is not None:
        #    b, _ = self.compute_cost(y_b, y, mask)
        #    cost = cost - b

        # make sure the cost is considered as a constant
        cost = cost.detach()
        #print('(METRICS)detached cost requires grad:', cost.requires_grad)

        # cost shape : b n
        if self.lam is None:
            lam = 1 / score.size(2)  # lam = 1 / num_nodes
        else:
            lam = self.lam
        if self.variance_reduced:
            # compute surrogate loss
            # First term below: Elementwise multiply node-level cost with node-level score.
            # Sedon term below: Multiply node-level cost with the sum of all node-level scores.
            val = cost * score + lam * cost * score.sum(-2, keepdims=True) # 128,1,30,1 * 128,1,30,1(score per node) + 128,1,30,1 * 128,1,1,1 (score per graph/sum of node scores)
            #print('(METRICS)score.sum(-2, keepdims=True) shape', score.sum(-2, keepdims=True).shape) # 128,1,1,1
            #print('(METRICS)Val shape', val.shape) # 128,1,30,1
            #print('(METRICS)surrogate loss requires grad:', val.requires_grad)
        else:
            # compute standard loss
            #val = cost * score.sum(-2, keepdims=True) # 128,1,30,1 * 128,(score per graph/sum of node scores) > node-level cost * graph-level score/sum of all node-level scores
            #val = cost * score
            #print('(METRICS)score.sum(-2, keepdims=True) shape', score.sum(-2, keepdims=True).shape) # 128,1,1,1
            #print('(METRICS)Val shape', val.shape) # 128,1,30,1
            #print('(METRICS)standard loss requires grad:', val.requires_grad)
            cost_sum_graph = cost.sum(-2, keepdims=True)
            score_sum_graph = score.sum(-2, keepdims=True)
            val = cost_sum_graph * score_sum_graph # 128,1,1,1
            #print('(METRICS)val_graph shape', val_graph.shape) # 128,1,1,1
            #print('Loss per graph after summing node-wise losses:', val.sum(-2, keepdims=True))
            #print('Loss per Graph directly calculated:', val_graph)
            #print('Are the above two equal?', torch.allclose(val.sum(-2, keepdims=True), val_graph, rtol=1e-5, atol=1e-8))
            

        self.value += val.sum() # Sum of all values in the val tensor(128,1,30,1). Ie, sum of 128*30 values(batch_count*node_count). 
        self.numel += mask.sum() # Sum of all valid elements without any missing values.

        #print('Loss per batch after summing node-wise losses over all batches:', self.value)
        #print('Loss per batch after summing graph-wise losses over all batches:', val_graph.sum())

        #print('MAE per batch after summing node-wise losses over all batches:', self.value/self.numel)
        #print('MAE per batch after summing graph-wise losses over all batches:', val_graph.sum()/self.numel)
        

        #print('(METRICS)Value(sum of Val):', self.value)
        #print('(METRICS)Numel(sum of Mask):', self.numel)

    def compute(self):
        # Averages the sum stored inside the value variable with the total number of valid elements in the dataset.
        if self.numel > 0:
            #print('(METRICS)Value(sum of Val) divided by Numel(Sum of Mask)', self.value / self.numel)
            #print('(METRICS)Value(sum of Val) divided by Numel(Sum of Mask) shape', (self.value / self.numel).shape)
            return self.value / self.numel
        return self.value
