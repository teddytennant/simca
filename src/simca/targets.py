"""Regression targets for the two generative families."""

from __future__ import annotations

from jax import Array


def psi_diffusion(eps: Array, sigma: Array | float) -> Array:
    """Denoising-score-matching target, -eps / sigma."""
    return -eps / sigma


def psi_flow(
    x0: Array,
    eps: Array,
    alpha_dot: Array | float,
    sigma_dot: Array | float,
) -> Array:
    """Conditional-flow-matching target, alpha_dot x0 + sigma_dot eps.

    On the linear path this is eps - x0.
    """
    return alpha_dot * x0 + sigma_dot * eps
