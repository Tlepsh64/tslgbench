# -*- coding: utf-8 -*-
"""
IMLE sampler for sparse graph edge selection, adapted from torch-imle
(https://github.com/uclnlp/torch-imle), MIT License, Copyright (c) Niepert, Minervini, Franceschi.
Implements "Implicit MLE: Backpropagating Through Discrete Exponential Family
Distributions" (Niepert, Minervini, Franceschi, NeurIPS 2021), specialized here
to top-k edge sampling ("BES") for graph structure learning.
"""

import torch
from torch import nn
from torch.autograd import Function

from lib.nn.graph_samplers.imle.noise import SumOfGammaNoiseDistribution


class BESIMLEFunction(Function):
    _last_stats = None
    edges_to_sample = 5  # Number of edges to sample per node

    @staticmethod
    def forward(ctx, scores, lambda_val, noise_temp, nb_samples=1):
        """
        scores: [batch_size, n_nodes, n_nodes] - graph logits
        Returns: sampled adjacency matrices with exactly k edges per node
        """
        batch_size, n_nodes, _ = scores.shape
        device = scores.device
        top_k = BESIMLEFunction.edges_to_sample

        # Sample noise
        noise_distribution = SumOfGammaNoiseDistribution(k=5, nb_iterations=100, device=device)
        noise = noise_distribution.sample(scores.shape) * noise_temp

        # Perturb scores
        perturbed_scores = scores + noise

        # GUMBEL-TOP-K: Sample exactly k edges per node
        # Find top-k indices for each row [batch_size, n_nodes, k]
        topk_values, topk_indices = torch.topk(perturbed_scores, k=top_k, dim=-1)

        # Create binary adjacency matrix using scatter
        samples = torch.zeros_like(scores)
        samples.scatter_(dim=-1, index=topk_indices,
                        src=torch.ones_like(topk_indices, dtype=torch.float))

        print('BESIMLEFUNCTION top-K MAP SAMPLES SHAPE:', samples.shape)

        # Verify we have exactly k ones per row
        ones_per_row = samples.sum(dim=-1)
        assert torch.all(ones_per_row == top_k), f"Expected exactly {top_k} ones per row"

        # Compute statistics
        with torch.no_grad():
            # Edge probabilities (using softmax for k-subset)
            log_probs = torch.log_softmax(perturbed_scores, dim=-1)
            selected_log_probs = torch.gather(log_probs, dim=-1, index=topk_indices)
            avg_log_prob = selected_log_probs.mean().item()

            density = samples.mean().item()
            expected_density = top_k / n_nodes
            noise_magnitude = noise.abs().mean().item()

        stats = {
            'avg_log_prob': avg_log_prob,
            'density': density,
            'expected_density': expected_density,
            'noise_magnitude': noise_magnitude,
            'k': top_k
        }

        BESIMLEFunction._last_stats = stats

        ctx.saved = {
            'scores': scores,
            'noise': noise,
            'samples': samples,
            'lambda_val': lambda_val,
            'noise_temp': noise_temp,
            'nb_samples': nb_samples,
        }

        print(f"I-MLE Forward: Sparsity={density:.4f} (expected={expected_density:.4f}), "
              f"Avg log prob={avg_log_prob:.4f}")

        return samples

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved
        scores, noise, samples = saved['scores'], saved['noise'], saved['samples']
        lambda_val, noise_temp = saved['lambda_val'], saved['noise_temp']
        device = scores.device
        top_k = BESIMLEFunction.edges_to_sample

        # I-MLE: Compute target distribution parameters
        target_scores = scores - lambda_val * grad_output

        # Sample from target distribution with same noise
        perturbed_target_scores = target_scores + noise * noise_temp

        # GUMBEL-TOP-K for target
        target_topk_values, target_topk_indices = torch.topk(perturbed_target_scores, k=top_k, dim=-1)

        # Create target samples (same pattern as forward)
        target_samples = torch.zeros_like(scores)
        target_samples.scatter_(dim=-1, index=target_topk_indices,
                               src=torch.ones_like(target_topk_indices, dtype=torch.float))

        # I-MLE gradient estimate
        gradient = (samples - target_samples)

        # Compute backward statistics
        with torch.no_grad():
            adj_grad_norm = grad_output.norm().item() if grad_output is not None else 0.0
            imle_grad_norm = gradient.norm().item()

            # Overlap analysis
            overlap_mask = (samples * target_samples).sum(dim=-1)  # [batch_size, n_nodes]
            overlap_per_node = overlap_mask.float() / top_k
            avg_overlap = overlap_per_node.mean().item()

            # Edge changes
            changes_per_row = (samples != target_samples).sum(dim=-1).float()
            avg_changes = changes_per_row.mean().item()

            gradient_ratio = imle_grad_norm / (adj_grad_norm + 1e-8)

        # Update stats
        if BESIMLEFunction._last_stats is not None:
            BESIMLEFunction._last_stats.update({
                'adj_grad_norm': adj_grad_norm,
                'imle_grad_norm': imle_grad_norm,
                'gradient_ratio': gradient_ratio,
                'avg_overlap': avg_overlap,
                'avg_changes': avg_changes
            })

        print(f"I-MLE Backward: Overlap={avg_overlap:.2%}, Changes={avg_changes:.1f}/row, "
              f"Grad ratio={gradient_ratio:.4f}")

        return gradient, None, None, None


def imle_bes_solver(scores, lambda_val=1.0, noise_temp=1.0, nb_samples=1):
    return BESIMLEFunction.apply(scores, lambda_val, noise_temp, nb_samples)


class IMLESampler(nn.Module):
    def __init__(self, lambda_val=10.0, noise_temp=1.0, nb_samples=1):
        super().__init__()
        self.lambda_val = lambda_val
        self.noise_temp = noise_temp
        self.nb_samples = nb_samples
        self._forward_stats = None

    def forward(self, scores, noise_temp=None, **kwargs):
        if noise_temp is None:
            noise_temp = self.noise_temp

        # Use the proper I-MLE function
        sample = imle_bes_solver(scores, self.lambda_val, self.noise_temp, self.nb_samples)

        self._forward_stats = getattr(BESIMLEFunction, '_last_stats', None)

        # For I-MLE, we don't compute log-likelihood
        return sample, None

    def mode(self, scores, tau):
        return torch.where(torch.sigmoid(scores / tau) > .5, 1., 0.)

    def get_stats(self):
        return self._forward_stats
