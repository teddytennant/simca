"""2 by 2 normal equations for the guidance weights."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

RIDGE = 1e-8


def solve_guidance_weights(
    u: Array,
    f: Array,
    psi: Array,
    ridge: float = RIDGE,
) -> Array:
    """Solve min_w ||w_u u + w_f f - psi||^2 via the normal equations.

    u, f, and psi share a shape that includes a batch axis and the data
    dimension. Inner products sum over every axis. A ridge of ``ridge`` is
    added to H only when H is numerically singular (collinear fields). A
    well-conditioned problem is solved unregularized.
    """
    uu = jnp.sum(u * u)
    uf = jnp.sum(u * f)
    ff = jnp.sum(f * f)
    up = jnp.sum(u * psi)
    fp = jnp.sum(f * psi)
    gram = jnp.array([[uu, uf], [uf, ff]])
    target = jnp.array([up, fp])
    evals = jnp.linalg.eigvalsh(gram)
    scale = jnp.maximum(jnp.max(jnp.abs(evals)), jnp.asarray(1.0, dtype=gram.dtype))
    singular = evals[0] < jnp.asarray(1e-4, dtype=gram.dtype) * scale
    eye = jnp.eye(2, dtype=gram.dtype)
    gram_used = gram + jnp.where(singular, jnp.asarray(ridge, dtype=gram.dtype), 0.0) * eye
    return jnp.linalg.solve(gram_used, target)
