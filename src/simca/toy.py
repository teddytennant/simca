"""Gaussian linear inverse problem with a closed-form unconditional field."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from simca.schedules import flow_linear, vp_cosine


@dataclass(frozen=True)
class GaussianProblem:
    """x0 ~ N(mean, covariance), y = A x0 + N(0, sigma_m^2 I)."""

    mean: Array
    covariance: Array
    A: Array
    sigma_m: float

    @property
    def d(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_obs(self) -> int:
        return int(self.A.shape[0])

    def sample(self, key: Array, n: int) -> tuple[Array, Array]:
        key_x, key_n = jr.split(key)
        chol = jnp.linalg.cholesky(self.covariance)
        x0 = self.mean + jr.normal(key_x, (n, self.d)) @ chol.T
        noise = self.sigma_m * jr.normal(key_n, (n, self.n_obs))
        y = x0 @ self.A.T + noise
        return x0, y

    def score(self, x: Array, t: Array | float) -> Array:
        """Marginal score of x_t = alpha x0 + sigma eps under the cosine schedule.

        x is a single state of shape (d,).
        """
        alpha, sigma = vp_cosine(t)
        cov_t = (alpha ** 2) * self.covariance + (sigma ** 2) * jnp.eye(self.d)
        return -jnp.linalg.solve(cov_t, x - alpha * self.mean)

    def velocity(self, x: Array, t: Array | float) -> Array:
        """Conditional velocity E[eps - x0 | x_t] on the linear flow path.

        x is a single state of shape (d,).
        """
        alpha, sigma = flow_linear(t)
        cov_t = (alpha ** 2) * self.covariance + (sigma ** 2) * jnp.eye(self.d)
        cross = alpha * self.covariance
        x0_cond = self.mean + cross @ jnp.linalg.solve(cov_t, x - alpha * self.mean)
        safe_sigma = jnp.where(jnp.abs(sigma) < 1e-6, jnp.ones_like(sigma), sigma)
        raw = (x - x0_cond) / safe_sigma
        return jnp.where(jnp.abs(sigma) < 1e-6, jnp.zeros_like(x), raw)

    def posterior_mean(self, y: Array) -> Array:
        """Kalman update for E[x0 | y]. y is (m,) or (N, m)."""
        eye = jnp.eye(self.n_obs)
        innov_cov = self.A @ self.covariance @ self.A.T + (self.sigma_m ** 2) * eye
        cross = self.covariance @ self.A.T
        if y.ndim == 1:
            innov = y - self.A @ self.mean
            return self.mean + cross @ jnp.linalg.solve(innov_cov, innov)
        innov = y - self.mean @ self.A.T
        coef = jnp.linalg.solve(innov_cov, innov.T).T
        return self.mean + coef @ cross.T


def gaussian_linear_problem(
    key: Array,
    d: int = 8,
    n_obs: int = 4,
    sigma_m: float = 0.05,
) -> GaussianProblem:
    """Random SPD prior and a random linear measurement operator."""
    key_mean, key_cov, key_op = jr.split(key, 3)
    mean = jr.normal(key_mean, (d,))
    raw = jr.normal(key_cov, (d, d))
    covariance = raw @ raw.T / d + 0.3 * jnp.eye(d)
    operator = jr.normal(key_op, (n_obs, d)) / jnp.sqrt(d)
    return GaussianProblem(mean, covariance, operator, sigma_m)


def squared_measurement_residual(y: Array, x0_hat: Array, operator: Array) -> Array:
    """Mean over the batch of ||y - A x0_hat||^2."""
    resid = y - x0_hat @ jnp.asarray(operator).T
    return jnp.mean(jnp.sum(resid * resid, axis=-1))
