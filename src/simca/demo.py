"""Run the synthetic weight check and the Gaussian toy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from simca.calibrate import (
    calibrate_simulation_based,
    calibrate_simulation_free,
    make_x0_hat_fn,
)
from simca.least_squares import solve_guidance_weights
from simca.schedules import time_grid, vp_cosine
from simca.toy import gaussian_linear_problem, squared_measurement_residual
from simca.tweedie import measurement_consistency, x0_hat_from_score


def synthetic_weights(key: Array) -> tuple[Array, Array]:
    """psi = 1.5 u - 0.4 f plus tiny noise. Weights should land near that pair."""
    key_u, key_f, key_n = jr.split(key, 3)
    u = jr.normal(key_u, (64, 8))
    f = jr.normal(key_f, (64, 8))
    noise = 1e-3 * jr.normal(key_n, u.shape)
    psi = 1.5 * u + (-0.4) * f + noise
    return solve_guidance_weights(u, f, psi), jnp.array([1.5, -0.4])


def main() -> None:
    recovered, target = synthetic_weights(jr.PRNGKey(0))
    print(
        "synthetic least squares: target "
        f"[{float(target[0]):.4f}, {float(target[1]):.4f}], recovered "
        f"[{float(recovered[0]):.4f}, {float(recovered[1]):.4f}]"
    )

    problem = gaussian_linear_problem(jr.PRNGKey(1), d=8, n_obs=4, sigma_m=0.05)
    x0, y = problem.sample(jr.PRNGKey(2), 256)
    times = jnp.array([0.25, 0.5, 0.75])
    weights = calibrate_simulation_free(
        x0, y, times, problem.score, problem.A, jr.PRNGKey(3), schedule="vp"
    )
    print("simulation-free weights (t=0.25, 0.5, 0.75), columns w_u, w_f:")
    for t, w in zip(times, weights):
        print(f"  t={float(t):.2f}  w=[{float(w[0]):.4f}, {float(w[1]):.4f}]")

    t = jnp.asarray(0.5)
    w = weights[1]
    eps = jr.normal(jr.PRNGKey(4), x0.shape)
    alpha, sigma = vp_cosine(t)
    state = alpha * x0 + sigma * eps
    u = jax.vmap(lambda z: problem.score(z, t))(state)
    f = measurement_consistency(state, y, t, problem.A, make_x0_hat_fn(problem.score, "vp"))
    unguided = x0_hat_from_score(state, u, alpha, sigma)
    guided = x0_hat_from_score(state, w[0] * u + w[1] * f, alpha, sigma)
    resid_u = squared_measurement_residual(y, unguided, problem.A)
    resid_g = squared_measurement_residual(y, guided, problem.A)
    mse_u = jnp.mean(jnp.sum((unguided - x0) ** 2, axis=-1))
    mse_g = jnp.mean(jnp.sum((guided - x0) ** 2, axis=-1))
    kalman = problem.posterior_mean(y)
    mse_k = jnp.mean(jnp.sum((kalman - x0) ** 2, axis=-1))
    mse_prior = jnp.mean(jnp.sum((problem.mean - x0) ** 2, axis=-1))
    print(
        "t=0.5 measurement residual: "
        f"unguided {float(resid_u):.4f}, guided {float(resid_g):.4f}"
    )
    print(
        "t=0.5 reconstruction mse: "
        f"unguided {float(mse_u):.4f}, guided {float(mse_g):.4f}, "
        f"kalman E[x0|y] {float(mse_k):.4f}, prior mean {float(mse_prior):.4f}"
    )

    sim_times = time_grid(8)
    sim_x0, sim_y = problem.sample(jr.PRNGKey(5), 32)
    sim_w = calibrate_simulation_based(
        sim_x0,
        sim_y,
        sim_times,
        problem.score,
        problem.A,
        jr.PRNGKey(6),
        schedule="vp",
    )
    print(
        "simulation-based K=8: shape "
        f"{tuple(sim_w.shape)}, all finite {bool(jnp.all(jnp.isfinite(sim_w)))}, "
        f"first w=[{float(sim_w[0, 0]):.4f}, {float(sim_w[0, 1]):.4f}], "
        f"last w=[{float(sim_w[-1, 0]):.4f}, {float(sim_w[-1, 1]):.4f}]"
    )


if __name__ == "__main__":
    main()
