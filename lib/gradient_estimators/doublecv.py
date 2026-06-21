import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from einops import rearrange


class DoubleCVEstimator(nn.Module):
    """
    Double Control Variates gradient estimator for discrete latent variables.

    Based on: "Double Control Variates for Gradient Estimation in Discrete Latent Variable Models"
    by Titsias & Shi (AISTATS 2022). Reference implementation:
    https://github.com/thjashin/double-cv (MIT License, Copyright (c) Titsias, Shi).

    The estimator computes:
        g = (1/K) Σ_k [(f(x_k) + α·b_k) - baseline_k] · s(x_k) - α · global_correction

    Where:
        - b_k = (1/(K-1) Σ_{j≠k} ∇f(x_j))^T · (x_k - μ)  (per-sample control variate)
        - s(x) = x - μ  (Bernoulli score function, computed via autodiff of log q)
        - global_correction = E[s(x)(x-μ)^T] · mean_grad = μ(1-μ) · mean_grad
        - α is learned by minimizing ||g||²

    Args:
        mode: Control variate mode. One of:
            - 'none': Pure RLOO (no control variates)
            - 'bk_only': Only b_k in objective, no b_j in baseline
            - 'bj_only': No b_k in objective, only b_j in baseline
            - 'full': Both b_k and b_j (full Double-CV)
        gradient_level: Granularity of control variates. One of:
            - 'graph_level': One scalar CV per graph
            - 'node_level': One CV per node
            - 'batch_level': One scalar CV for entire batch
        alpha_lr: Learning rate for α parameter (default: 1e-3)
    """

    def __init__(self, mode='full', gradient_level='graph_level', alpha_lr=1e-3):
        super().__init__()
        self.mode = mode
        self.gradient_level = gradient_level
        self.alpha_lr = alpha_lr

        # α parameter for control variate weighting (initialized at 0 as per paper)
        self._alpha = nn.Parameter(torch.tensor(0.0))

        # Determine which control variates to use
        self.use_bk = mode in ['bk_only', 'full']  # b_k in objective
        self.use_bj = mode in ['bj_only', 'full']  # b_j in baseline

    @property
    def alpha(self):
        return self._alpha

    def forward(self, costs, grads, scores, edge_logits, differences, mask, batch_size=128):
        """
        Compute the Double-CV gradient estimator.

        Args:
            costs: (k, batch, 1, nodes, 1) - f(x_k) prediction losses for each sample
            grads: (k, batch, nodes, nodes) - ∇f(x_k) gradients w.r.t. edge weights
            scores: (k, batch, nodes) - log q(x_k) for REINFORCE surrogate loss
            edge_logits: (batch, nodes, nodes) - μ = sigmoid(θ), the edge probabilities
            differences: (k, batch, nodes, nodes) - x_k - μ for each sample
            mask: Loss mask for valid elements
            batch_size: Batch size (default: 128)

        Returns:
            graph_loss: The surrogate loss for backpropagation
            pred_loss: Mean prediction loss across samples
            metrics: Dictionary of monitoring metrics
        """
        k = costs.shape[0]
        valid_element_count = mask.sum()

        if k > 1:
            # Compute leave-one-out sums, control variates, objectives, and baselines
            f_xj_sum = self._compute_leave_one_out(costs, k)
            b_tensors, bj_sum = self._compute_control_variates(grads, differences, k, batch_size)
            objectives, baselines, correlations = self._build_objectives_baselines(
                costs, b_tensors, bj_sum, f_xj_sum, k, batch_size
            )

            # Compute graph loss (surrogate loss for REINFORCE)
            graph_loss = self._compute_graph_loss(objectives, baselines, scores, k, valid_element_count)

            # Apply global correction if using b_k
            if self.use_bk:
                graph_loss = self._apply_global_correction(graph_loss, grads, edge_logits, valid_element_count)

            # Update α by minimizing ||g||²
            self._update_alpha(graph_loss)

            # Compute variance metrics
            metrics = self._compute_variance_metrics(objectives, baselines, scores, mask, valid_element_count, k)
            metrics.update(correlations)
        else:
            # Single sample case - no leave-one-out possible
            objectives = self._aggregate_costs(costs[0], batch_size)
            baselines = torch.zeros_like(objectives)
            graph_loss = self._compute_single_sample_loss(objectives, scores[0], valid_element_count)
            metrics = {
                'advantage_variance': torch.tensor(0.0),
                'estimator_variance': torch.tensor(0.0),
                'mean_advantage': torch.tensor(0.0),
                'snr': torch.tensor(0.0)
            }

        # Compute mean prediction loss
        mean_losses = [sample.sum() / valid_element_count for sample in costs]
        pred_loss = torch.stack(mean_losses).mean()

        return graph_loss, pred_loss, metrics

    def _compute_leave_one_out(self, costs, k):
        """Compute leave-one-out sums for f(x_j)."""
        leave_one_out = []
        for i in range(k):
            indices = [j for j in range(k) if j != i]
            summed = costs[indices].detach().sum(dim=0)
            leave_one_out.append(summed)
        return torch.stack(leave_one_out)

    def _compute_control_variates(self, grads, differences, k, batch_size):
        """
        Compute b_k control variates: b_k = (1/(K-1) * Σ_{j≠k} ∇f(x_j))^T · (x_k - μ)
        Also compute leave-one-out sums for b_j.
        """
        b_tensors = []
        for i in range(k):
            indices = [j for j in range(k) if j != i]
            # Mean of gradients for j != k
            bk_mean_grad = grads[indices].detach().sum(dim=0) / (k - 1)
            bk_difference = differences[i].detach()

            # Compute dot product based on gradient level
            if self.gradient_level == 'graph_level':
                bk = (bk_mean_grad * bk_difference).sum(dim=(1, 2))  # (batch,)
            elif self.gradient_level == 'node_level':
                bk = (bk_mean_grad * bk_difference).sum(dim=2)  # (batch, nodes)
            elif self.gradient_level == 'batch_level':
                bk = (bk_mean_grad * bk_difference).sum()  # scalar

            b_tensors.append(bk.detach())

        b_tensors = torch.stack(b_tensors)

        # Compute leave-one-out sums for b_j
        bj_sum = []
        for i in range(k):
            indices = [j for j in range(k) if j != i]
            summed = b_tensors[indices].detach().sum(dim=0)
            bj_sum.append(summed)
        bj_sum = torch.stack(bj_sum)

        return b_tensors, bj_sum

    def _build_objectives_baselines(self, costs, b_tensors, bj_sum, f_xj_sum, k, batch_size):
        """Build objectives (f(x_k) + α*b_k) and baselines for each sample."""
        objectives = []
        baselines = []
        correlation_fxk_global = []
        correlation_objective_baseline = []

        for i in range(k):
            # Aggregate costs to appropriate level
            f_xk = self._aggregate_costs(costs[i], batch_size)
            f_xj = self._aggregate_costs(f_xj_sum[i], batch_size)

            # Build objective: f(x_k) + α * b_k (if using b_k)
            if self.use_bk:
                global_cv = self._reshape_cv(b_tensors[i], f_xk.shape, batch_size)
                objective = f_xk.detach() + self._alpha * global_cv
            else:
                objective = f_xk.detach()
                global_cv = None

            # Build baseline: (1/(K-1)) * Σ_{j≠k} [f(x_j) + α * b_j] (if using b_j)
            if self.use_bj:
                bj_cv = self._reshape_cv(bj_sum[i], f_xj.shape, batch_size)
                baseline = (f_xj.detach() / (k - 1)) + (self._alpha * bj_cv) / (k - 1)
            else:
                baseline = f_xj.detach() / (k - 1)

            objectives.append(objective)
            baselines.append(baseline)

            # Compute correlations for monitoring
            if self.use_bk and global_cv is not None:
                corr = self._compute_correlation(f_xk, self._alpha * global_cv)
                if corr is not None:
                    correlation_fxk_global.append(corr)

            if self.mode == 'none':
                corr = self._compute_correlation(f_xk, baseline)
                if corr is not None:
                    correlation_objective_baseline.append(corr)

        correlations = {}
        if correlation_fxk_global:
            correlations['corr_fxk_globalcv'] = np.mean(correlation_fxk_global)
        if correlation_objective_baseline:
            correlations['corr_objective_baseline'] = np.mean(correlation_objective_baseline)

        return torch.stack(objectives), torch.stack(baselines), correlations

    def _aggregate_costs(self, cost, batch_size):
        """Aggregate costs to the appropriate gradient level."""
        if self.gradient_level == 'graph_level':
            return cost.sum(dim=2, keepdim=True)  # (batch, 1, 1, 1)
        elif self.gradient_level == 'node_level':
            return cost  # (batch, 1, nodes, 1)
        elif self.gradient_level == 'batch_level':
            return cost.sum()  # scalar

    def _reshape_cv(self, cv, target_shape, batch_size):
        """Reshape control variate to match target shape."""
        if self.gradient_level == 'graph_level':
            return cv.view(batch_size, 1, 1, 1).expand(target_shape)
        elif self.gradient_level == 'node_level':
            return cv.view(batch_size, 1, -1, 1).expand(target_shape)
        elif self.gradient_level == 'batch_level':
            return cv  # already scalar

    def _compute_graph_loss(self, objectives, baselines, scores, k, valid_element_count):
        """Compute the surrogate loss for REINFORCE gradient."""
        graph_losses = []

        for i in range(k):
            cost = objectives[i] - baselines[i]
            score = scores[i]
            score = rearrange(score, 'b n -> b 1 n 1')

            if self.gradient_level == 'graph_level':
                val = cost * score.sum(-2, keepdims=True)
            elif self.gradient_level == 'node_level':
                val = cost * score.sum(-2, keepdims=True)
            elif self.gradient_level == 'batch_level':
                val = cost * score.sum()

            graph_loss_i = val.sum() / valid_element_count
            graph_losses.append(graph_loss_i)

        return torch.stack(graph_losses).mean(dim=0)

    def _compute_single_sample_loss(self, objective, score, valid_element_count):
        """Compute loss for single sample case (k=1)."""
        score = rearrange(score, 'b n -> b 1 n 1')

        if self.gradient_level == 'graph_level':
            val = objective * score.sum(-2, keepdims=True)
        elif self.gradient_level == 'node_level':
            val = objective * score.sum(-2, keepdims=True)
        elif self.gradient_level == 'batch_level':
            val = objective * score.sum()

        return val.sum() / valid_element_count

    def _apply_global_correction(self, graph_loss, grads, edge_logits, valid_element_count):
        """Apply global correction term: α * E[s(x)(x-μ)^T] * mean_grad."""
        mean_grads = grads.mean(dim=0)
        expected_value_term = edge_logits.detach() * (1 - edge_logits.detach())
        correction = (mean_grads.detach() * expected_value_term).sum() / valid_element_count
        correction = self._alpha * correction
        return graph_loss - correction

    def _update_alpha(self, graph_loss):
        """Update α by computing gradient of ||g||²."""
        alpha_loss = graph_loss ** 2
        if alpha_loss.requires_grad:
            alpha_grad = torch.autograd.grad(
                alpha_loss, self._alpha,
                retain_graph=True,
                allow_unused=True
            )[0]
            if alpha_grad is not None:
                self._alpha.grad = alpha_grad

    def _compute_correlation(self, tensor1, tensor2):
        """Compute Pearson correlation between two tensors."""
        try:
            if self.gradient_level == 'batch_level':
                # Can't compute correlation for scalars
                return None

            flat1 = tensor1.flatten().cpu().detach().numpy()
            flat2 = tensor2.flatten().cpu().detach().numpy()

            if len(flat1) < 2 or np.std(flat1) == 0 or np.std(flat2) == 0:
                return None

            corr, p_value = pearsonr(flat1, flat2)
            if p_value >= 0.05:
                print(f'p-value is {p_value}, correlation is not statistically significant.')
            return corr
        except Exception:
            return None

    def _compute_variance_metrics(self, objectives, baselines, scores, mask, valid_element_count, k):
        """Compute variance metrics for monitoring."""
        with torch.no_grad():
            advantages = objectives.detach() - baselines.detach()

            # 1. Variance of advantages (unweighted)
            if self.gradient_level == 'graph_level':
                var_per_element = torch.var(advantages, dim=0, unbiased=True)
                advantage_variance = var_per_element.sum() / valid_element_count
            elif self.gradient_level == 'node_level':
                var_per_element = torch.var(advantages, dim=0, unbiased=True)
                advantage_variance = (var_per_element * mask).sum() / valid_element_count
            elif self.gradient_level == 'batch_level':
                advantage_variance = torch.var(advantages, unbiased=True)

            # 2. Score-weighted variance (actual gradient estimator variance)
            score_reshaped = rearrange(scores.detach(), 'k b n -> k b 1 n 1')

            if self.gradient_level == 'graph_level':
                score_weights = score_reshaped.sum(dim=-2, keepdim=True)
                weighted_advantages = advantages * score_weights
                weighted_var = torch.var(weighted_advantages, dim=0, unbiased=True)
                estimator_variance = weighted_var.sum() / valid_element_count
            elif self.gradient_level == 'node_level':
                score_weights = score_reshaped.sum(dim=-2, keepdim=True)
                weighted_advantages = advantages * score_weights.expand_as(advantages)
                weighted_var = torch.var(weighted_advantages, dim=0, unbiased=True)
                estimator_variance = (weighted_var * mask).sum() / valid_element_count
            elif self.gradient_level == 'batch_level':
                score_weights = score_reshaped.sum(dim=(1, 2, 3, 4))
                weighted_advantages = advantages * score_weights
                estimator_variance = torch.var(weighted_advantages, unbiased=True)

            # 3. Mean advantage (should be close to zero for unbiased estimator)
            mean_advantage = advantages.mean()

            # 4. Signal-to-noise ratio
            if advantage_variance > 0:
                snr = torch.abs(mean_advantage) / torch.sqrt(advantage_variance)
            else:
                snr = torch.tensor(float('inf'))

        return {
            'advantage_variance': advantage_variance,
            'estimator_variance': estimator_variance,
            'mean_advantage': mean_advantage,
            'snr': snr
        }
