"""Invariants for the guidance-weight calibration."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from simca.calibrate import (
    calibrate_simulation_based,
    calibrate_simulation_free,
    fields_at_state,
    make_x0_hat_fn,
)
from simca.least_squares import solve_guidance_weights
from simca.sampler import euler_step, probability_flow_drift
from simca.schedules import flow_linear, flow_linear_dots, time_grid, vp_cosine, vp_cosine_dots
from simca.targets import psi_flow
from simca.toy import gaussian_linear_problem, squared_measurement_residual
from simca.tweedie import (
    measurement_consistency,
    x0_hat_from_score,
    x0_hat_from_velocity,
)


def test_normal_equations_recover_known_weights():
    key_u, key_f = jr.split(jr.PRNGKey(0))
    u = jr.normal(key_u, (48, 6))
    f = jr.normal(key_f, (48, 6))
    psi = 1.5 * u + (-0.4) * f
    weights = solve_guidance_weights(u, f, psi)
    assert weights.shape == (2,)
    assert jnp.allclose(weights, jnp.array([1.5, -0.4]), atol=1e-6)


def test_ridge_on_collinear_fields_is_finite():
    u = jr.normal(jr.PRNGKey(1), (20, 4))
    f = 2.5 * u
    psi = 0.3 * u
    weights = solve_guidance_weights(u, f, psi)
    assert weights.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(weights)))
    # The consistent equation is w_u + 2.5 w_f = 0.3. Ridge should not
    # abandon that relation.
    predicted = weights[0] + 2.5 * weights[1]
    assert jnp.allclose(predicted, jnp.asarray(0.3), atol=1e-4)


def test_simulation_free_beats_ignore_measurements():
    problem = gaussian_linear_problem(jr.PRNGKey(0), d=8, n_obs=4, sigma_m=0.05)
    t = jnp.asarray(0.5)
    x0, y = problem.sample(jr.PRNGKey(1), 128)
    weights = calibrate_simulation_free(
        x0,
        y,
        jnp.asarray([t]),
        problem.score,
        problem.A,
        jr.PRNGKey(2),
        schedule="vp",
    )
    assert weights.shape == (1, 2)
    assert bool(jnp.all(jnp.isfinite(weights)))

    # Held-out forward states at the same interior time.
    x0_e, y_e = problem.sample(jr.PRNGKey(3), 128)
    eps = jr.normal(jr.PRNGKey(4), x0_e.shape)
    alpha, sigma = vp_cosine(t)
    state = alpha * x0_e + sigma * eps
    u = jax.vmap(lambda z: problem.score(z, t))(state)
    f = measurement_consistency(
        state, y_e, t, problem.A, make_x0_hat_fn(problem.score, "vp")
    )
    w = weights[0]
    unguided = x0_hat_from_score(state, u, alpha, sigma)
    guided = x0_hat_from_score(state, w[0] * u + w[1] * f, alpha, sigma)
    resid_u = squared_measurement_residual(y_e, unguided, problem.A)
    resid_g = squared_measurement_residual(y_e, guided, problem.A)
    assert bool(jnp.isfinite(resid_u) and jnp.isfinite(resid_g))
    assert float(resid_g) < float(resid_u)


def test_simulation_based_four_steps_are_finite():
    problem = gaussian_linear_problem(jr.PRNGKey(0), d=8, n_obs=4, sigma_m=0.05)
    x0, y = problem.sample(jr.PRNGKey(1), 16)
    times = time_grid(4)
    weights = calibrate_simulation_based(
        x0,
        y,
        times,
        problem.score,
        problem.A,
        jr.PRNGKey(2),
        schedule="vp",
    )
    assert weights.shape == (4, 2)
    assert bool(jnp.all(jnp.isfinite(weights)))
    assert float(times[0]) == pytest.approx(1.0)
    assert float(times[-1]) > 0.0
    assert bool(jnp.all(jnp.diff(times) < 0.0))


def test_tweedie_score_recovers_clean_sample():
    t = jnp.asarray(0.3)
    alpha, sigma = vp_cosine(t)
    x0 = jr.normal(jr.PRNGKey(3), (5, 8))
    eps = jr.normal(jr.PRNGKey(4), (5, 8))
    state = alpha * x0 + sigma * eps
    score = -eps / sigma
    recovered = x0_hat_from_score(state, score, alpha, sigma)
    assert jnp.allclose(recovered, x0, atol=1e-6)


def test_flow_denoiser_recovers_clean_sample():
    t = jnp.asarray(0.4)
    alpha, sigma = flow_linear(t)
    x0 = jr.normal(jr.PRNGKey(5), (5, 8))
    eps = jr.normal(jr.PRNGKey(6), (5, 8))
    state = alpha * x0 + sigma * eps
    velocity = eps - x0
    recovered = x0_hat_from_velocity(state, velocity, sigma)
    assert jnp.allclose(recovered, x0, atol=1e-6)


def test_flow_least_squares_uses_cfm_target():
    t = jnp.asarray(0.35)
    alpha_dot, sigma_dot = flow_linear_dots(t)
    x0 = jr.normal(jr.PRNGKey(7), (32, 5))
    eps = jr.normal(jr.PRNGKey(8), (32, 5))
    psi = psi_flow(x0, eps, alpha_dot, sigma_dot)
    assert jnp.allclose(psi, -x0 + eps, atol=1e-6)

    key_u, key_f = jr.split(jr.PRNGKey(9))
    u = jr.normal(key_u, (32, 5))
    f = jr.normal(key_f, (32, 5))
    # Known weights in front of the flow target, not a diffusion score.
    mixed = 1.5 * u + (-0.4) * f
    # Rebuild a flow target that equals that mixture by using the identity
    # psi = eps - x0, with eps = x0 + mixed.
    flow_psi = psi_flow(x0, x0 + mixed, alpha_dot, sigma_dot)
    assert jnp.allclose(flow_psi, mixed, atol=1e-6)
    weights = solve_guidance_weights(u, f, flow_psi)
    assert jnp.allclose(weights, jnp.array([1.5, -0.4]), atol=1e-6)


def test_diffusion_drift_matches_conditional_path():
    t = jnp.asarray(0.4)
    x0 = jr.normal(jr.PRNGKey(10), (4, 8))
    eps = jr.normal(jr.PRNGKey(11), (4, 8))
    alpha, sigma = vp_cosine(t)
    state = alpha * x0 + sigma * eps
    score = -eps / sigma
    alpha_dot, sigma_dot = vp_cosine_dots(t)
    expected = alpha_dot * x0 + sigma_dot * eps
    got = probability_flow_drift(state, score, t)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_flow_euler_step_reaches_data():
    t = jnp.asarray(0.4)
    x0 = jr.normal(jr.PRNGKey(12), (4, 8))
    eps = jr.normal(jr.PRNGKey(13), (4, 8))
    alpha, sigma = flow_linear(t)
    state = alpha * x0 + sigma * eps
    velocity = eps - x0
    landed = euler_step(state, velocity, t, 0.0, schedule="flow")
    assert jnp.allclose(landed, x0, atol=1e-6)


def test_simulation_free_matches_fields_at_state():
    problem = gaussian_linear_problem(jr.PRNGKey(14), d=8, n_obs=4, sigma_m=0.05)
    x0, y = problem.sample(jr.PRNGKey(15), 24)
    t = jnp.asarray(0.6)
    key = jr.PRNGKey(16)
    weights = calibrate_simulation_free(
        x0, y, jnp.asarray([t]), problem.score, problem.A, key, schedule="vp"
    )
    step_key = jr.split(key, 1)[0]
    eps = jr.normal(step_key, x0.shape)
    alpha, sigma = vp_cosine(t)
    state = alpha * x0 + sigma * eps
    direct, _, _ = fields_at_state(
        state, x0, y, t, problem.score, problem.A, "vp", eps
    )
    assert jnp.allclose(weights[0], direct, atol=1e-6)


def test_callable_operator_matches_matrix():
    problem = gaussian_linear_problem(jr.PRNGKey(17), d=8, n_obs=4, sigma_m=0.05)
    x0, y = problem.sample(jr.PRNGKey(18), 8)
    t = jnp.asarray(0.45)
    eps = jr.normal(jr.PRNGKey(19), x0.shape)
    alpha, sigma = vp_cosine(t)
    state = alpha * x0 + sigma * eps
    denoiser = make_x0_hat_fn(problem.score, "vp")

    def apply(z):
        return problem.A @ z

    f_matrix = measurement_consistency(state, y, t, problem.A, denoiser)
    f_callable = measurement_consistency(state, y, t, apply, denoiser)
    assert jnp.allclose(f_matrix, f_callable, atol=1e-5)


def test_kalman_beats_prior_mean():
    problem = gaussian_linear_problem(jr.PRNGKey(20), d=8, n_obs=4, sigma_m=0.05)
    x0, y = problem.sample(jr.PRNGKey(21), 64)
    post = problem.posterior_mean(y)
    mse_post = jnp.mean(jnp.sum((post - x0) ** 2, axis=-1))
    mse_prior = jnp.mean(jnp.sum((problem.mean - x0) ** 2, axis=-1))
    assert float(mse_post) < float(mse_prior)


def test_flow_calibration_is_finite_and_not_the_diffusion_path():
    problem = gaussian_linear_problem(jr.PRNGKey(22), d=8, n_obs=4, sigma_m=0.05)
    x0, y = problem.sample(jr.PRNGKey(23), 32)
    times = jnp.array([0.8, 0.5, 0.2])
    key = jr.PRNGKey(24)
    flow_w = calibrate_simulation_free(
        x0, y, times, problem.velocity, problem.A, key, schedule="flow"
    )
    diff_w = calibrate_simulation_free(
        x0, y, times, problem.score, problem.A, key, schedule="vp"
    )
    assert flow_w.shape == (3, 2)
    assert bool(jnp.all(jnp.isfinite(flow_w)))
    assert not bool(jnp.allclose(flow_w, diff_w, atol=1e-3))
