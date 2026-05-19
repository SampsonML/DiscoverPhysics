"""Analytic reference solutions used to validate the N-body simulator.

These functions are intentionally kept out of the core simulator so that the
simulator is free of any test / verification code.  They are imported by the
integrator-comparison notebook to provide ground-truth trajectories.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

# ── Kepler equation solver ─────────────────────────────────────────────────


def solve_kepler(mean_anomaly, eccentricity, max_iter: int = 64):
    """Solve ``E - e sin E = M`` for the eccentric anomaly ``E``.

    Uses Newton iteration with a fixed iteration count (chosen so that the
    function is fully traceable / JIT-compilable).  64 iterations gets to
    machine precision for any ``e < ~0.95``.
    """
    M = jnp.asarray(mean_anomaly)
    e = eccentricity
    # E0 = M + e sin M is a robust starting guess.
    E = M + e * jnp.sin(M)

    def body(_, E):
        f = E - e * jnp.sin(E) - M
        fp = 1.0 - e * jnp.cos(E)
        return E - f / fp

    return jax.lax.fori_loop(0, max_iter, body, E)


# ── Two-body Kepler orbit ──────────────────────────────────────────────────


def kepler_two_body_solution(
    times,
    M_central,
    m_orbiter,
    semi_major,
    eccentricity,
    G: float = 1.0,
    t_periapsis: float = 0.0,
):
    """Closed-form bound Kepler orbit in 2D.

    The returned trajectory is the *relative* orbit (orbiter relative to
    central body): the orbiter starts at periapsis on the +x axis at time
    ``t_periapsis``.  When ``m_orbiter << M_central`` the central body is
    nearly fixed and the simulator's absolute-frame trajectory matches; for
    finite mass ratios, compare to ``r_orbiter - r_central``.

    Returns
    -------
    pos : (T, 2) array
        Position of the orbiter relative to the central body.
    vel : (T, 2) array
        Velocity of the orbiter relative to the central body.
    period : float
        Orbital period.
    """
    a = semi_major
    e = eccentricity
    mu = G * (M_central + m_orbiter)
    n = jnp.sqrt(mu / a**3)  # mean motion
    period = 2 * jnp.pi / n

    M = n * (jnp.asarray(times) - t_periapsis)
    E = solve_kepler(M, e)

    cosE = jnp.cos(E)
    sinE = jnp.sin(E)
    sqrt1me2 = jnp.sqrt(1.0 - e**2)

    x = a * (cosE - e)
    y = a * sqrt1me2 * sinE

    # dE/dt = n / (1 - e cos E).
    dEdt = n / (1.0 - e * cosE)
    vx = -a * sinE * dEdt
    vy = a * sqrt1me2 * cosE * dEdt

    pos = jnp.stack([x, y], axis=-1)
    vel = jnp.stack([vx, vy], axis=-1)
    return pos, vel, period


# ── Circular-orbit velocity for arbitrary central forces ───────────────────


def circular_orbit_velocity(
    force_law: Callable, M_central, m_orbiter, radius, q_central=None, q_orbiter=None
):
    """Tangential speed for a circular orbit at radius ``r`` (test-mass limit).

    Solves ``F(r) = m v^2 / r`` → ``v = sqrt(F(r) r / m)`` using whatever
    ``force_law`` is supplied.  ``force_law`` follows the simulator
    convention ``force_law(r, q_i, q_j, m_i, m_j) -> F_mag``; when no
    explicit charges are given they default to the masses (so plain
    Newtonian gravity works without supplying ``q_*``).
    """
    if q_central is None:
        q_central = M_central
    if q_orbiter is None:
        q_orbiter = m_orbiter
    F = float(
        force_law(
            jnp.asarray(radius),
            jnp.asarray(q_central),
            jnp.asarray(q_orbiter),
            jnp.asarray(M_central),
            jnp.asarray(m_orbiter),
        )
    )
    return float(jnp.sqrt(F * radius / m_orbiter))
