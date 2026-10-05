"""Simulation-free and simulation-based guidance-weight calibration."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from simca.least_squares import solve_guidance_weights
from simca.sampler import euler_step
from simca.schedules import alpha_sigma, alpha_sigma_dots
from simca.targets import psi_diffusion, psi_flow
from simca.tweedie import (
    Operator,
    measurement_consistency,
    x0_hat_from_score,
    x0_hat_from_velocity,
)

FieldFn = Callable[[Array, Array | float], Array]


def regression_target(
    schedule: str,
    x0: Array,
    eps: Array,
    t: Array | float,
) -> Array:
    """psi_t for diffusion (score target) or flow matching (velocity target)."""
    if schedule == "vp":
        _, sigma = alpha_sigma(schedule, t)
        return psi_diffusion(eps, sigma)
    if schedule == "flow":
        alpha_dot, sigma_dot = alpha_sigma_dots(schedule, t)
        return psi_flow(x0, eps, alpha_dot, sigma_dot)
    raise ValueError(f"unknown schedule {schedule!r}; expected 'vp' or 'flow'")


def make_x0_hat_fn(u_theta: FieldFn, schedule: str) -> Callable:
    """Denoiser induced by the unconditional field at a single state."""

    def x0_hat(x: Array, t: Array | float) -> Array:
        field = u_theta(x, t)
        if schedule == "vp":
            alpha, sigma = alpha_sigma(schedule, t)
            return x0_hat_from_score(x, field, alpha, sigma)
        if schedule == "flow":
            _, sigma = alpha_sigma(schedule, t)
            return x0_hat_from_velocity(x, field, sigma)
        raise ValueError(f"unknown schedule {schedule!r}; expected 'vp' or 'flow'")

    return x0_hat


def _check_batch(x0: Array, y: Array) -> None:
    if x0.ndim != 2 or y.ndim != 2:
        raise ValueError("x0 and y must be batches with shapes (N, d) and (N, m)")
    if x0.shape[0] != y.shape[0]:
        raise ValueError("x0 and y must have the same batch size")


def fields_at_state(
    x: Array,
    x0: Array,
    y: Array,
    t: Array | float,
    u_theta: FieldFn,
    operator: Operator,
    schedule: str,
    eps: Array,
) -> tuple[Array, Array, Array]:
    """Weights and the two fields at a known state.

    ``eps`` is the noise used in the regression target. On the simulation-free
    path it is the noise that built x. On the simulation-based path it is
    recovered from the sampler state and the paired clean sample.
    """
    u = jax.vmap(lambda z: u_theta(z, t))(x)
    f = measurement_consistency(x, y, t, operator, make_x0_hat_fn(u_theta, schedule))
    psi = regression_target(schedule, x0, eps, t)
    weights = solve_guidance_weights(u, f, psi)
    return weights, u, f


def calibrate_simulation_free(
    x0: Array,
    y: Array,
    times: Array,
    u_theta: FieldFn,
    operator: Operator,
    key: Array,
    schedule: str = "vp",
) -> Array:
    """Independent least-squares weights at each time.

    At each t, draw eps ~ N(0, I) and set x_t = alpha_t x0 + sigma_t eps.
    There is no trajectory. Returns weights with shape (K, 2).
    """
    _check_batch(x0, y)
    times = jnp.asarray(times)
    if times.ndim != 1 or times.shape[0] == 0:
        raise ValueError("times must be a nonempty 1d grid")
    step_keys = jr.split(key, times.shape[0])
    weights = []
    for t, step_key in zip(times, step_keys):
        _, sigma = alpha_sigma(schedule, t)
        if float(sigma) <= 1e-8:
            raise ValueError("sigma(t) is zero; do not calibrate at t=0")
        eps = jr.normal(step_key, x0.shape)
        alpha, sigma = alpha_sigma(schedule, t)
        state = alpha * x0 + sigma * eps
        step_weights, _, _ = fields_at_state(
            state, x0, y, t, u_theta, operator, schedule, eps
        )
        weights.append(step_weights)
    return jnp.stack(weights, axis=0)


def calibrate_simulation_based(
    x0: Array,
    y: Array,
    times: Array,
    u_theta: FieldFn,
    operator: Operator,
    key: Array,
    schedule: str = "vp",
) -> Array:
    """Least-squares weights along one Euler trajectory per sample.

    The state is initialized at t_K = 1 from N(0, I). Times must be decreasing
    and must not include 0, where sigma vanishes. After the weights at t_k are
    solved, the state is stepped to the next smaller time with
    Phi(x, w_u u + w_f f, t). The paper's listing writes that update as
    x at t_{k+1}. The sampler equation steps toward data, and so does this
    function. The last step lands at t = 0 and is not itself a calibration time.
    Returns weights with shape (K, 2).
    """
    _check_batch(x0, y)
    times = jnp.asarray(times)
    if times.ndim != 1 or times.shape[0] == 0:
        raise ValueError("times must be a nonempty 1d grid")
    key, key_init = jr.split(key)
    state = jr.normal(key_init, x0.shape)
    weights = []
    n_times = int(times.shape[0])
    for k in range(n_times):
        t = times[k]
        if k + 1 < n_times:
            t_next = times[k + 1]
        else:
            t_next = jnp.zeros((), dtype=times.dtype)
        alpha, sigma = alpha_sigma(schedule, t)
        if float(sigma) <= 1e-8:
            raise ValueError("sigma(t) is zero; drop t=0 from the calibration grid")
        eps = (state - alpha * x0) / sigma
        step_weights, u, f = fields_at_state(
            state, x0, y, t, u_theta, operator, schedule, eps
        )
        guided = step_weights[0] * u + step_weights[1] * f
        state = euler_step(state, guided, t, t_next, schedule=schedule)
        weights.append(step_weights)
    return jnp.stack(weights, axis=0)
