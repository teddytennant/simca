"""Tweedie denoiser and the measurement-consistency gradient."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

# cos(pi/2) is a few ulps away from 0. Treat that boundary as alpha = 0 so
# the 0/0 Tweedie quotient stays finite. Interior times are far above this.
ALPHA_EPS = 1e-6

Operator = Array | Callable[[Array], Array]
X0Hat = Callable[[Array, Array | float], Array]


def x0_hat_from_score(
    x: Array,
    score: Array,
    alpha: Array | float,
    sigma: Array | float,
) -> Array:
    """Tweedie estimate (x + sigma^2 score) / alpha for a score network.

    At alpha = 0 the quotient is undefined. The value returned there is 0 and
    does not depend on x, so its gradient is 0.
    """
    alpha = jnp.asarray(alpha)
    sigma = jnp.asarray(sigma)
    numer = x + (sigma ** 2) * score
    small = jnp.abs(alpha) < ALPHA_EPS
    safe_alpha = jnp.where(small, jnp.ones_like(alpha), alpha)
    raw = numer / safe_alpha
    return jnp.where(small, jnp.zeros_like(x), raw)


def x0_hat_from_velocity(x: Array, velocity: Array, sigma: Array | float) -> Array:
    """Denoiser x - sigma * v on the linear flow path."""
    return x - jnp.asarray(sigma) * velocity


def apply_operator(operator: Operator, x: Array) -> Array:
    """Apply a matrix (m, d) or a JAX-traceable map x -> y."""
    if callable(operator):
        return operator(x)
    return jnp.asarray(operator) @ x


def measurement_consistency(
    x: Array,
    y: Array,
    t: Array | float,
    operator: Operator,
    x0_hat_fn: X0Hat,
) -> Array:
    """Gradient of ||y - A(x0_hat(x, t))||^2 with respect to x.

    This is the raw gradient. The guidance weight absorbs scale, including the
    factor of 2 from differentiating a square and any missing proportionality
    constant in the likelihood.
    """
    single = x.ndim == 1
    if single:
        x = x[None, :]
        y = y[None, :]

    def one(xi: Array, yi: Array) -> Array:
        def loss(z: Array) -> Array:
            pred = apply_operator(operator, x0_hat_fn(z, t))
            resid = yi - pred
            return jnp.sum(resid * resid)

        return jax.grad(loss)(xi)

    grad = jax.vmap(one)(x, y)
    return grad[0] if single else grad
