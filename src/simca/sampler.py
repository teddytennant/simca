"""Euler step on the probability-flow ODE."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from simca.schedules import alpha_sigma, alpha_sigma_dots
from simca.tweedie import ALPHA_EPS


def probability_flow_drift(x: Array, score: Array, t: Array | float) -> Array:
    """dx/dt = alpha_dot x0_hat + sigma_dot eps_hat, with eps_hat = -sigma score.

    Equivalent to the probability-flow drift of the variance-preserving path.
    At alpha = 0 the Tweedie quotient is undefined. The pure-noise limit used
    there drops the alpha_dot x0_hat term, which is zero for this schedule
    because sigma_dot(1) = 0.
    """
    alpha, sigma = alpha_sigma("vp", t)
    alpha_dot, sigma_dot = alpha_sigma_dots("vp", t)
    small = jnp.abs(alpha) < ALPHA_EPS
    safe_alpha = jnp.where(small, jnp.ones_like(alpha), alpha)
    x0_hat = (x + (sigma ** 2) * score) / safe_alpha
    eps_hat = -sigma * score
    drift = alpha_dot * x0_hat + sigma_dot * eps_hat
    drift_limit = sigma_dot * eps_hat
    return jnp.where(small, drift_limit, drift)


def euler_step(
    x: Array,
    field: Array,
    t: Array | float,
    t_next: Array | float,
    schedule: str = "vp",
) -> Array:
    """One Euler step from t toward t_next.

    For ``vp``, ``field`` is treated as a score and converted to a
    probability-flow drift. For ``flow``, ``field`` is already a velocity.
    Sampling toward data uses t_next < t.
    """
    dt = jnp.asarray(t_next) - jnp.asarray(t)
    if schedule == "vp":
        drift = probability_flow_drift(x, field, t)
    elif schedule == "flow":
        drift = field
    else:
        raise ValueError(f"unknown schedule {schedule!r}; expected 'vp' or 'flow'")
    return x + dt * drift
