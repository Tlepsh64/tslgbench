# -*- coding: utf-8 -*-
"""
AIMLE sampler for sparse graph edge selection, adapted from torch-adaptive-imle
(https://github.com/EdinburghNLP/torch-adaptive-imle), MIT License,
Copyright (c) Minervini, Franceschi, Niepert.
Implements "Adaptive Perturbation-Based Gradient Estimation for Discrete Latent
Variable Models" (Minervini, Franceschi, Niepert, AAAI 2023), specialized here
to top-k / MST-based edge sampling ("BES") for graph structure learning.
"""

import torch
from torch import nn
from torch.autograd import Function

from lib.nn.graph_samplers.imle.noise import SumOfGammaNoiseDistribution
from lib.nn.graph_samplers.imle.target_dist import AdaptiveTargetDistribution
from lib.nn.graph_samplers.imle.solver import kruskal_mst_batched


class BESAIMLEFunction(Function):
    _last_stats = None
    edges_to_sample = 5  # Number of edges to sample per node (for topk solver)
    target_distribution = AdaptiveTargetDistribution()  # Will hold AdaptiveTargetDistribution instance
    symmetric_perturbation = False  # Central difference flag (class-level default)
    map_solver = "topk"  # MAP solver: "topk" or "kruskal"

    @staticmethod
    def _apply_map_solver(perturbed_scores, map_solver, top_k):
        """
        Apply the specified MAP solver to perturbed scores.

        Args:
            perturbed_scores: [batch_size, nb_samples, n_nodes, n_nodes] or [batch_size, n_nodes, n_nodes]
            map_solver: "topk" or "kruskal"
            top_k: Number of edges per node (for topk solver)

        Returns:
            samples: Binary adjacency matrices of same shape as perturbed_scores
        """
        if map_solver == "topk":
            # GUMBEL-TOP-K: Sample exactly k edges per node
            topk_values, topk_indices = torch.topk(perturbed_scores, k=top_k, dim=-1)
            samples = torch.zeros_like(perturbed_scores)
            samples.scatter_(dim=-1, index=topk_indices,
                            src=torch.ones_like(topk_indices, dtype=torch.float))
        elif map_solver == "kruskal":
            # Kruskal's MST algorithm
            original_shape = perturbed_scores.shape
            if len(original_shape) == 4:
                # Reshape [batch, nb_samples, n_nodes, n_nodes] -> [batch * nb_samples, n_nodes, n_nodes]
                batch_size, nb_samples, n_nodes, _ = original_shape
                perturbed_flat = perturbed_scores.reshape(-1, n_nodes, n_nodes)
                samples_flat = kruskal_mst_batched(perturbed_flat)
                samples = samples_flat.reshape(original_shape)
            else:
                # Shape is [batch_size, n_nodes, n_nodes]
                samples = kruskal_mst_batched(perturbed_scores)
        else:
            raise ValueError(f"Unknown MAP solver: {map_solver}. Choose 'topk' or 'kruskal'.")

        return samples

    @staticmethod
    def forward(ctx, scores, noise_temp, nb_samples=1, nb_marginal_samples=1):
        """
        scores: [batch_size, n_nodes, n_nodes] - graph logits
        nb_samples: number of independent noise samples
        nb_marginal_samples: number of samples to average for marginal estimation
        Returns: sampled adjacency matrices
        """
        batch_size, n_nodes, _ = scores.shape
        device = scores.device
        top_k = BESAIMLEFunction.edges_to_sample
        map_solver = BESAIMLEFunction.map_solver
        nb_total_samples = nb_samples * nb_marginal_samples

        # Create shape for multiple samples: [batch_size, nb_total_samples, n_nodes, n_nodes]
        perturbed_scores_shape = [batch_size, nb_total_samples, n_nodes, n_nodes]

        # Sample noise for all samples
        noise_distribution = SumOfGammaNoiseDistribution(k=top_k, nb_iterations=100, device=device)
        noise = noise_distribution.sample(torch.Size(perturbed_scores_shape)) * noise_temp

        # Expand scores to match: [batch_size, nb_total_samples, n_nodes, n_nodes]
        scores_expanded = scores.unsqueeze(1).expand(perturbed_scores_shape)

        # Perturb scores
        perturbed_scores = scores_expanded + noise

        # Apply the MAP solver
        samples = BESAIMLEFunction._apply_map_solver(perturbed_scores, map_solver, top_k)

        # Compute statistics
        with torch.no_grad():
            density = samples.mean().item()
            if map_solver == "topk":
                expected_density = top_k / n_nodes
                log_probs = torch.log_softmax(perturbed_scores, dim=-1)
                topk_indices = torch.topk(perturbed_scores, k=top_k, dim=-1)[1]
                selected_log_probs = torch.gather(log_probs, dim=-1, index=topk_indices)
                avg_log_prob = selected_log_probs.mean().item()
            else:
                # For Kruskal's MST, expected density is (n-1) edges in a tree / n^2 total possible
                expected_density = (n_nodes - 1) / (n_nodes * n_nodes)
                avg_log_prob = 0.0  # Not applicable for MST
            noise_magnitude = noise.abs().mean().item()

        stats = {
            'avg_log_prob': avg_log_prob,
            'density': density,
            'expected_density': expected_density,
            'noise_magnitude': noise_magnitude,
            'k': top_k,
            'nb_samples': nb_samples,
            'nb_marginal_samples': nb_marginal_samples,
            'map_solver': map_solver
        }

        BESAIMLEFunction._last_stats = stats

        # Save for backward
        ctx.saved = {
            'scores': scores,
            'noise': noise,
            'samples': samples,
            'noise_temp': noise_temp,
            'nb_samples': nb_samples,
            'nb_marginal_samples': nb_marginal_samples,
        }

        # Return averaged samples across all MC samples for forward pass
        # Shape: [batch_size, n_nodes, n_nodes]
        samples_mean = samples.mean(dim=1)

        print(f"AIMLE Forward ({map_solver}): Sparsity={density:.4f} (expected={expected_density:.4f}), "
              f"Avg log prob={avg_log_prob:.4f}, nb_samples={nb_samples}, nb_marginal={nb_marginal_samples}")

        return samples_mean

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved
        scores = saved['scores']
        noise = saved['noise']
        samples = saved['samples']  # Shape: [batch_size, nb_total_samples, n_nodes, n_nodes]
        noise_temp = saved['noise_temp']
        nb_samples = saved['nb_samples']
        nb_marginal_samples = saved['nb_marginal_samples']

        device = scores.device
        top_k = BESAIMLEFunction.edges_to_sample
        map_solver = BESAIMLEFunction.map_solver
        symmetric_perturbation = BESAIMLEFunction.symmetric_perturbation
        nb_total_samples = nb_samples * nb_marginal_samples

        batch_size, n_nodes, _ = scores.shape

        # Get the adaptive target distribution
        target_dist = BESAIMLEFunction.target_distribution
        if target_dist is None:
            raise ValueError("Must call BESAIMLEFunction.set_target_distribution() before use")

        # Expand scores and grad_output to match samples shape
        # [batch_size, nb_total_samples, n_nodes, n_nodes]
        scores_expanded = scores.unsqueeze(1).expand_as(samples)
        grad_output_expanded = grad_output.unsqueeze(1).expand_as(samples)

        # Flatten for processing: [batch_size * nb_total_samples, n_nodes, n_nodes]
        scores_flat = scores_expanded.reshape(-1, n_nodes, n_nodes)
        grad_output_flat = grad_output_expanded.reshape(-1, n_nodes, n_nodes)
        noise_flat = noise.reshape(-1, n_nodes, n_nodes)
        samples_flat = samples.reshape(-1, n_nodes, n_nodes)

        # 1. Compute target parameters: θ'_R = θ - λ∇f
        target_params_r = target_dist.params(scores_flat, grad_output_flat, symmetric=False)

        # 2. For central difference, also compute θ'_L = θ + λ∇f
        if symmetric_perturbation:
            target_params_l = target_dist.params(scores_flat, grad_output_flat, symmetric=True)

        # 3. Sample from target distribution(s) with same noise
        #perturbed_target_r = target_params_r + noise_flat * noise_temp # MINOR ERROR: Noise flat is already scaled by noise_temp. No need to multiply with temp again.
        perturbed_target_r = target_params_r + noise_flat

        # Apply MAP solver for target (right side)
        target_samples_r = BESAIMLEFunction._apply_map_solver(perturbed_target_r, map_solver, top_k)

        if symmetric_perturbation:
            # Apply MAP solver for target (left side)
            #perturbed_target_l = target_params_l + noise_flat * noise_temp
            perturbed_target_l = target_params_l + noise_flat
            target_samples_l = BESAIMLEFunction._apply_map_solver(perturbed_target_l, map_solver, top_k)
            # Central difference: g = (z_L - z_R)
            raw_gradient_flat = target_samples_l - target_samples_r
        else:
            # Forward difference: g = (z - z_R) where z is the original sample
            raw_gradient_flat = samples_flat - target_samples_r

        # Reshape back to [batch_size, nb_total_samples, n_nodes, n_nodes]
        raw_gradient = raw_gradient_flat.reshape(batch_size, nb_total_samples, n_nodes, n_nodes)
        target_samples_r_reshaped = target_samples_r.reshape(batch_size, nb_total_samples, n_nodes, n_nodes)

        # Average over marginal samples if nb_marginal_samples > 1
        if nb_marginal_samples > 1:
            # Reshape to [batch_size, nb_samples, nb_marginal_samples, n_nodes, n_nodes]
            raw_gradient = raw_gradient.reshape(batch_size, nb_samples, nb_marginal_samples, n_nodes, n_nodes)
            # Average over marginal samples
            raw_gradient = raw_gradient.mean(dim=2)  # [batch_size, nb_samples, n_nodes, n_nodes]

        # Average over all samples to get final gradient estimate
        # Shape: [batch_size, n_nodes, n_nodes]
        """raw_gradient_mean = raw_gradient.mean(dim=1) # MINOR ERROR FIXED. Process first, then average the grads.

        print('RAW GRADIENT SHAPE:', raw_gradient_mean.shape)

        # 4. Let the adaptive target distribution process and normalize the gradient
        gradient = target_dist.process(scores, grad_output, raw_gradient_mean,
                                       symmetric_perturbation=symmetric_perturbation)"""

        gradient = target_dist.process(scores_expanded, grad_output_expanded, raw_gradient, symmetric_perturbation=symmetric_perturbation)
        gradient = gradient.mean(dim=1)

        # Compute backward statistics
        with torch.no_grad():
            adj_grad_norm = grad_output.norm().item() if grad_output is not None else 0.0
            imle_grad_norm = gradient.norm().item()

            # Overlap analysis (use first sample for stats)
            samples_first = samples[:, 0, :, :]  # [batch_size, n_nodes, n_nodes]
            target_first = target_samples_r_reshaped[:, 0, :, :]
            overlap_mask = (samples_first * target_first).sum(dim=-1)
            overlap_per_node = overlap_mask.float() / top_k
            avg_overlap = overlap_per_node.mean().item()

            # Edge changes
            changes_per_row = (samples_first != target_first).sum(dim=-1).float()
            avg_changes = changes_per_row.mean().item()

            gradient_ratio = imle_grad_norm / (adj_grad_norm + 1e-8)

            # Track adaptive parameters
            beta_val = target_dist.beta
            pm = target_dist._perturbation_magnitude(scores, grad_output)
            current_pm = pm.item() if isinstance(pm, torch.Tensor) else pm

        # Update stats
        if BESAIMLEFunction._last_stats is not None:
            BESAIMLEFunction._last_stats.update({
                'adj_grad_norm': adj_grad_norm,
                'imle_grad_norm': imle_grad_norm,
                'gradient_ratio': gradient_ratio,
                'avg_overlap': avg_overlap,
                'avg_changes': avg_changes,
                'beta': beta_val,
                'pm': current_pm,
                'grad_norm': target_dist.grad_norm,
                'symmetric': symmetric_perturbation
            })

        diff_type = "Central" if symmetric_perturbation else "Forward"
        print(f"AIMLE Backward ({diff_type}): β={beta_val:.4f}, λ(pm)={current_pm:.4f}, "
              f"Overlap={avg_overlap:.2%}, Grad norm={target_dist.grad_norm:.3f}")

        return gradient, None, None, None  # gradient, None for noise_temp, nb_samples, nb_marginal_samples


def aimle_bes_solver(scores, noise_temp=1.0, nb_samples=1, nb_marginal_samples=1):
    return BESAIMLEFunction.apply(scores, noise_temp, nb_samples, nb_marginal_samples)


class AIMLESampler(nn.Module):
    def __init__(self, noise_temp=1.0, nb_samples=1, nb_marginal_samples=1, symmetric_perturbation=False,
                 map_solver="topk", use_momentum=False, k=5):
        """
        AIMLE Sampler for k-subset selection.

        Args:
            noise_temp: Temperature for noise perturbation
            nb_samples: Number of independent noise samples for gradient estimation
            nb_marginal_samples: Number of samples to average for marginal estimation (variance reduction)
            symmetric_perturbation: If True, use central difference; if False, use forward difference
            map_solver: MAP solver to use - "topk" for k-subset selection, "kruskal" for MST
            use_momentum: If True, use momentum for adaptive β updates
            k: Number of edges per node for topk solver; also shapes the SumOfGamma noise distribution
        """
        super().__init__()
        self.noise_temp = noise_temp
        self.nb_samples = nb_samples
        self.nb_marginal_samples = nb_marginal_samples
        self.symmetric_perturbation = symmetric_perturbation
        self.map_solver = map_solver
        self.use_momentum = use_momentum
        self._forward_stats = None

        # Set the class-level flags
        BESAIMLEFunction.edges_to_sample = k
        BESAIMLEFunction.symmetric_perturbation = symmetric_perturbation
        BESAIMLEFunction.map_solver = map_solver

        # Update the target distribution with use_momentum setting
        BESAIMLEFunction.target_distribution.use_momentum = use_momentum

    def forward(self, scores, noise_temp=None, **kwargs):
        if noise_temp is None:
            noise_temp = self.noise_temp

        # Ensure class-level settings are correct (in case they were changed)
        BESAIMLEFunction.symmetric_perturbation = self.symmetric_perturbation
        BESAIMLEFunction.map_solver = self.map_solver
        BESAIMLEFunction.target_distribution.use_momentum = self.use_momentum

        # Use the AIMLE function with all parameters
        sample = aimle_bes_solver(scores, noise_temp, self.nb_samples, self.nb_marginal_samples)

        self._forward_stats = getattr(BESAIMLEFunction, '_last_stats', None)

        return sample, None

    def mode(self, scores, tau):
        return torch.where(torch.sigmoid(scores / tau) > .5, 1., 0.)

    def get_stats(self):
        return self._forward_stats

    def set_symmetric_perturbation(self, symmetric: bool):
        """Enable or disable central difference (symmetric perturbation)."""
        self.symmetric_perturbation = symmetric
        BESAIMLEFunction.symmetric_perturbation = symmetric

    def set_map_solver(self, map_solver: str):
        """Set the MAP solver to use ('topk' or 'kruskal')."""
        self.map_solver = map_solver
        BESAIMLEFunction.map_solver = map_solver

    def set_use_momentum(self, use_momentum: bool):
        """Enable or disable momentum for adaptive β updates."""
        self.use_momentum = use_momentum
        BESAIMLEFunction.target_distribution.use_momentum = use_momentum
