# -*- coding: utf-8 -*-
"""
Target distributions for AIMLE, adapted from torch-adaptive-imle
(https://github.com/EdinburghNLP/torch-adaptive-imle), MIT License,
Copyright (c) Minervini, Franceschi, Niepert.
"""

import torch

from torch import Tensor
from abc import ABC, abstractmethod
from typing import Optional


class BaseTargetDistribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def params(self,
               theta: Tensor,
               dy: Optional[Tensor],
               _is_minimization: bool = False) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def process(self,
                theta: Tensor,
                dy: Tensor,
                gradient: Tensor) -> Tensor:
        return gradient


class AdaptiveTargetDistribution(BaseTargetDistribution):
    """
    Adaptive Target Distribution for AIMLE.

    This class implements the adaptive β (perturbation magnitude) mechanism from the AIMLE paper.
    It tracks gradient sparsity and adjusts β (which determines λ) to maintain a target gradient density.

    The relationship is: λ = β * ||θ||₂ / ||∇f||₂ (Equation 8 in the paper)
    """
    def __init__(self,
                 initial_beta: float = 1.0,  # Written as zero in the paper's pseudocode section.
                 initial_grad_norm: float = 1.0,
                 # Pitch: the initial default hyperparams lead to very stable results,
                 # competitive with manually tuned ones -- E.g. try with 1e-3 for this hyperparam
                 beta_update_step: float = 0.001,
                 beta_update_momentum: float = 0.9,
                 use_momentum: bool = False,
                 grad_norm_decay_rate: float = 0.9,
                 target_norm: float = 1.0):
        super().__init__()
        self.beta = initial_beta
        self.grad_norm = initial_grad_norm
        self.beta_update_step = beta_update_step
        self.beta_update_momentum = beta_update_momentum
        self.use_momentum = use_momentum
        self.previous_beta_update = 0.0
        self.grad_norm_decay_rate = grad_norm_decay_rate
        self.target_norm = target_norm

    def _perturbation_magnitude(self,
                                theta: Tensor,
                                dy: Optional[Tensor]):
        """Compute λ = β * ||θ||₂ / ||∇f||₂ (Equation 8)"""
        norm_dy = torch.linalg.norm(dy).item() if dy is not None else 1.0
        return 0.0 if norm_dy <= 0.0 else self.beta * (torch.linalg.norm(theta) / norm_dy)

    def params(self,
               theta: Tensor,
               dy: Optional[Tensor],
               symmetric: bool = False,
               _is_minimization: bool = False) -> Tensor:
        """
        Compute target distribution parameters.

        Args:
            theta: Original parameters
            dy: Downstream gradients
            symmetric: If True, compute θ + λ∇f (for central difference left side)
                      If False, compute θ - λ∇f (standard forward difference / central difference right side)
        """
        pm = self._perturbation_magnitude(theta, dy)
        if symmetric:
            # For central difference: θ'_L = θ + λ∇f
            theta_prime = theta + pm * (dy if dy is not None else 0.0)
        else:
            # Standard: θ'_R = θ - λ∇f
            theta_prime = theta - pm * (dy if dy is not None else 0.0)
        return theta_prime

    def process(self,
                theta: Tensor,
                dy: Tensor,
                gradient_tensor: Tensor,
                symmetric_perturbation: bool = False) -> Tensor:
        """
        Process and normalize the gradient, updating adaptive parameters.

        Args:
            theta: Original parameters
            dy: Downstream gradients
            gradient_tensor: Raw gradient (z_L - z_R)
            symmetric_perturbation: If True, divide by 2 for central difference
        """
        pm = self._perturbation_magnitude(theta, dy)

        # We compute an exponentially decaying sum of the gradient norms
        grad_nnz = torch.count_nonzero(gradient_tensor).float()
        # Dynamically compute the number of gradient elements from tensor shape
        # For shape [batch, n_nodes, n_nodes] or [batch, nb_samples, n_nodes, n_nodes]
        nb_gradients = gradient_tensor.numel()
        batch_size, nb_samples, n_nodes = gradient_tensor.shape[0], gradient_tensor.shape[1], gradient_tensor.shape[2]
        denominator = batch_size * nb_samples # Minervini's method of normalization
        gradient_sparsity = grad_nnz / denominator
        #denominator = gradient_tensor.numel() # Alternative to Minervini's method. More aggressive than above.
        #gradient_sparsity = grad_nnz / nb_gradients

        # Running estimate of the gradient norm (number of non-zero elements for every sample)
        self.grad_norm = self.grad_norm_decay_rate * self.grad_norm + (1.0 - self.grad_norm_decay_rate) * gradient_sparsity

        # If the gradient norm is lower than target, we increase beta; otherwise, we decrease beta.
        beta_update_ = (1.0 if self.grad_norm.item() < self.target_norm else -1.0) * self.beta_update_step

        # Apply momentum if enabled
        if self.use_momentum:
            beta_update = (self.beta_update_momentum * self.previous_beta_update) + beta_update_
        else:
            beta_update = beta_update_

        # Enforcing β ≥ 0
        self.beta = max(self.beta + beta_update, 0.0)
        self.previous_beta_update = beta_update

        # Normalize gradient by perturbation magnitude
        res = gradient_tensor / (pm if pm > 0.0 else 1.0)

        # For central difference, divide by 2 as per Algorithm 1 in the paper
        if symmetric_perturbation:
            res = res / 2.0

        return res
