# -*- coding: utf-8 -*-

import torch

from torch import Tensor
from abc import ABC, abstractmethod

from typing import Optional

import logging

logger = logging.getLogger(__name__)


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


class TargetDistribution(BaseTargetDistribution):
    r"""
    Creates a generator of target distributions parameterized by :attr:`alpha` and :attr:`beta`.

    Example::

        >>> import torch
        >>> target_distribution = TargetDistribution(alpha=1.0, beta=1.0)
        >>> target_distribution.params(theta=torch.tensor([1.0]), dy=torch.tensor([1.0]))
        tensor([2.])

    Args:
        alpha (float): weight of the initial distribution parameters theta
        beta (float): weight of the downstream gradient dy
        do_gradient_scaling (bool): whether to scale the gradient by 1/λ or not
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 do_gradient_scaling: bool = False,
                 eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.do_gradient_scaling = do_gradient_scaling
        self.eps = eps

    def params(self,
               theta: Tensor,
               dy: Optional[Tensor],
               alpha: Optional[float] = None,
               beta: Optional[float] = None,
               _is_minimization: bool = False) -> Tensor:
        alpha_ = self.alpha if alpha is None else alpha
        beta_ = self.beta if beta is None else beta

        if _is_minimization is True:
            theta_prime = alpha_ * theta + beta_ * (dy if dy is not None else 0.0)
        else:
            theta_prime = alpha_ * theta - beta_ * (dy if dy is not None else 0.0)
        return theta_prime

    def process(self,
                theta: Tensor,
                dy: Tensor,
                gradient_3d: Tensor) -> Tensor:
        scaling_factor = max(self.beta, self.eps)
        res = (gradient_3d / scaling_factor) if self.do_gradient_scaling is True else gradient_3d
        return res


class AdaptiveTargetDistribution(BaseTargetDistribution):
    """
    Adaptive Target Distribution for AIMLE.

    This class implements the adaptive λ (perturbation magnitude) mechanism from the AIMLE paper.
    It tracks gradient sparsity and adjusts α (which determines λ) to maintain a target gradient density.

    Note: The paper uses β for the adaptive parameter, but this implementation uses α.
    The relationship is: λ = α * ||θ||₂ / ||∇f||₂ (Equation 8 in the paper)
    """
    def __init__(self,
                 initial_alpha: float = 0.0,  # Written as zero in the paper's pseudocode section.
                 initial_grad_norm: float = 1.0,
                 # Pitch: the initial default hyperparams lead to very stable results,
                 # competitive with manually tuned ones -- E.g. try with 1e-3 for this hyperparam
                 alpha_update_step: float = 0.001,
                 alpha_update_momentum: float = 0.9,
                 use_momentum: bool = False,
                 grad_norm_decay_rate: float = 0.9,
                 target_norm: float = 1.0):
        super().__init__()
        self.alpha = initial_alpha
        self.grad_norm = initial_grad_norm
        self.alpha_update_step = alpha_update_step
        self.alpha_update_momentum = alpha_update_momentum
        self.use_momentum = use_momentum
        self.previous_alpha_update = 0.0
        self.grad_norm_decay_rate = grad_norm_decay_rate
        self.target_norm = target_norm

    def _perturbation_magnitude(self,
                                theta: Tensor,
                                dy: Optional[Tensor]):
        """Compute λ = α * ||θ||₂ / ||∇f||₂ (Equation 8)"""
        norm_dy = torch.linalg.norm(dy).item() if dy is not None else 1.0
        return 0.0 if norm_dy <= 0.0 else self.alpha * (torch.linalg.norm(theta) / norm_dy)

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
            # Just Forward Difference: θ'_R = θ - λ∇f
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
            gradient_tensor: Raw gradient (z_L - z_R) or (z - z_R)
            symmetric_perturbation: If True, divide by 2 for central difference
        """
        pm = self._perturbation_magnitude(theta, dy)

        # We compute an exponentially decaying sum of the gradient norms
        grad_nnz = torch.count_nonzero(gradient_tensor).float()
        # Dynamically compute the number of gradient elements from tensor shape
        nb_gradients = gradient_tensor.numel()
        gradient_density = grad_nnz / nb_gradients

        print('SHAPE OF GRADIENT TENSOR:', gradient_tensor.shape, f'density={gradient_density:.4f}')

        # Running estimate of the gradient norm (number of non-zero elements for every sample)
        self.grad_norm = self.grad_norm_decay_rate * self.grad_norm + (1.0 - self.grad_norm_decay_rate) * gradient_density

        # If the gradient norm is lower than target, we increase alpha; otherwise, we decrease alpha.
        #alpha_update_ = (1.0 if self.grad_norm.item() < self.target_norm else - 1.0) * self.alpha_update_step
        density_difference = self.grad_norm.item() - self.target_norm
        if density_difference < 0:
            alpha_update_ = self.alpha_update_step
        else:
            alpha_update_ = -1 * self.alpha_update_step

        # Apply momentum if enabled
        if self.use_momentum:
            alpha_update = (self.alpha_update_momentum * self.previous_alpha_update) + alpha_update_
        else:
            alpha_update = alpha_update_

        # Enforcing α ≥ 0
        self.alpha = max(self.alpha + alpha_update, 0.0)
        self.previous_alpha_update = alpha_update

        # print(f'Gradient norm: {self.grad_norm:.5f}\talpha: {self.alpha:.5f}')

        # Normalize gradient by perturbation magnitude
        res = gradient_tensor / (pm if pm > 0.0 else 1.0)

        # For central difference, divide by 2 as per Algorithm 1 in the paper
        if symmetric_perturbation:
            res = res / 2.0

        return res