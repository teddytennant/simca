"""Noise schedules with t=0 at data and t=1 at noise."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def vp_cosine(t: Array | float) -> tuple[Array, Array]:
    """Variance-preserving cosine schedule, alpha^2 + sigma^2 = 1."""
    t = jnp.asarray(t)
    alpha = jnp.cos(0.5 * jnp.pi * t)
    sigma = jnp.sin(0.5 * jnp.pi * t)
    return alpha, sigma


def vp_cosine_dots(t: Array | float) -> tuple[Array, Array]:
    """Time derivatives of the variance-preserving cosine schedule."""
    alpha, sigma = vp_cosine(t)
    half_pi = 0.5 * jnp.pi
    return -half_pi * sigma, half_pi * alpha


def flow_linear(t: Array | float) -> tuple[Array, Array]:
    """Independent-coupling linear interpolant, alpha = 1 - t, sigma = t."""
    t = jnp.asarray(t)
    return 1.0 - t, t


def flow_linear_dots(t: Array | float) -> tuple[Array, Array]:
    """Time derivatives of the linear interpolant."""
    t = jnp.asarray(t)
    return jnp.ones_like(t) * -1.0, jnp.ones_like(t)


def alpha_sigma(schedule: str, t: Array | float) -> tuple[Array, Array]:
    if schedule == "vp":
        return vp_cosine(t)
    if schedule == "flow":
        return flow_linear(t)
    raise ValueError(f"unknown schedule {schedule!r}; expected 'vp' or 'flow'")


def alpha_sigma_dots(schedule: str, t: Array | float) -> tuple[Array, Array]:
    if schedule == "vp":
        return vp_cosine_dots(t)
    if schedule == "flow":
        return flow_linear_dots(t)
    raise ValueError(f"unknown schedule {schedule!r}; expected 'vp' or 'flow'")


def time_grid(num_steps: int) -> Array:
    """Decreasing grid t_K = 1, ..., t_1 = 1/K. Excludes t = 0."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    return jnp.linspace(1.0, 1.0 / num_steps, num_steps)
