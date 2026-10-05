"""Offline least-squares calibration of guidance weights."""

from simca.calibrate import (
    calibrate_simulation_based,
    calibrate_simulation_free,
    fields_at_state,
)
from simca.least_squares import RIDGE, solve_guidance_weights
from simca.sampler import euler_step, probability_flow_drift
from simca.schedules import (
    flow_linear,
    flow_linear_dots,
    time_grid,
    vp_cosine,
    vp_cosine_dots,
)
from simca.targets import psi_diffusion, psi_flow
from simca.tweedie import (
    measurement_consistency,
    x0_hat_from_score,
    x0_hat_from_velocity,
)
from simca.toy import GaussianProblem, gaussian_linear_problem

__version__ = "0.1.0"

__all__ = [
    "GaussianProblem",
    "RIDGE",
    "calibrate_simulation_based",
    "calibrate_simulation_free",
    "euler_step",
    "fields_at_state",
    "flow_linear",
    "flow_linear_dots",
    "gaussian_linear_problem",
    "measurement_consistency",
    "probability_flow_drift",
    "psi_diffusion",
    "psi_flow",
    "solve_guidance_weights",
    "time_grid",
    "vp_cosine",
    "vp_cosine_dots",
    "x0_hat_from_score",
    "x0_hat_from_velocity",
]
