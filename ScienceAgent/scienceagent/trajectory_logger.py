"""
TrajectoryLogger: streams experiment trajectories to a per-world CSV file.

One row per (experiment, time, particle). Each invocation of
`run_discovery.py` writes a unique `run_id`; rows append to the existing
file so the CSV accumulates across runs.

Columns
-------
run_id, experiment_id, round, source, time, particle_id, x, y, vx, vy,
p1, p2, ring_radius, initial_tangential_velocity

Per-world conventions
---------------------
- 2-particle worlds (gravity/yukawa/fractional/diffusion/wave): p1, p2 are
  filled; ring_radius and initial_tangential_velocity are blank. Particle 0
  is the fixed source at origin (in agent-relative coords), particle 1 is
  the mobile probe.
- circle: ring_radius and initial_tangential_velocity are filled; 11
  particles (0 = centre, 1-10 = ring).
- species: 6 particles, all four scalar params blank (initial state is
  fully encoded by the t=0 rows).
- three_species: 35 particles. Background (0-29) initial positions come
  from the output's `background_initial_positions`, with zero velocities.
  Probes (30-34) come from the experiment input.
- dark_matter: 25 agent-visible particles. Visible (0-19) positions come
  from `background_initial_positions`; visible velocities come from
  `executor._visible_velocities` (scaled by `visible_velocity_sign` if the
  evaluator flipped orbit direction). Probes (20-24) come from the input.

The first rows of every experiment are at time=0 with the reconstructed
initial state, even if the agent's measurement_times list does not include
0. This gives MSE-fitting code a clean handle on initial conditions.
"""

from __future__ import annotations

import csv
import os
import re
import datetime as _dt
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

COLUMNS: tuple[str, ...] = (
    "run_id",
    "experiment_id",
    "round",
    "source",
    "time",
    "particle_id",
    "x",
    "y",
    "vx",
    "vy",
    "p1",
    "p2",
    "ring_radius",
    "initial_tangential_velocity",
    "mass",
    "charge",
)


def make_run_id(model: str, when: Optional[_dt.datetime] = None) -> str:
    """Compact, filesystem-safe run identifier: <ISOtime>_<model-slug>."""
    when = when or _dt.datetime.now()
    stamp = when.strftime("%Y%m%dT%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    return f"{stamp}_{slug}"


class TrajectoryLogger:
    """Append per-experiment trajectory rows to results/trajectories/<world>.csv.

    Single instance per `run_discovery.py` invocation. Pass it into
    `DiscoveryAgent` (which calls `log_experiment` after each successful
    `executor.run`).
    """

    def __init__(
        self,
        world: str,
        executor,
        csv_path: os.PathLike | str,
        run_id: str,
    ):
        self.world = world
        self.executor = executor
        self.csv_path = Path(csv_path)
        self.run_id = run_id
        self._row_builder = _RowBuilders.get(world)
        if self._row_builder is None:
            raise ValueError(
                f"TrajectoryLogger has no row builder for world '{world}'. "
                f"Known worlds: {sorted(_RowBuilders)}"
            )

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._needs_header = (
            not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        )

    def log_experiment(
        self,
        round_num: int,
        source: str,
        exp_input: dict,
        exp_output: dict,
        exp_idx_in_round: int = 0,
    ) -> None:
        """Append all rows (t=0 + observed times) for one experiment."""
        if not isinstance(exp_input, dict) or not isinstance(exp_output, dict):
            return
        experiment_id = f"{self.run_id}__r{round_num}__e{exp_idx_in_round}"
        rows = self._row_builder(self.executor, exp_input, exp_output)
        if not rows:
            return

        for row in rows:
            row["run_id"] = self.run_id
            row["experiment_id"] = experiment_id
            row["round"] = round_num
            row["source"] = source

        self._write(rows)

    def _write(self, rows: Sequence[dict]) -> None:
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if self._needs_header:
                writer.writeheader()
                self._needs_header = False
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Per-world row builders. Each takes (executor, exp_input, exp_output) and
# returns a list of partially-populated row dicts (run-level fields are
# filled in by the logger). Particle ids are 0-indexed.


def _scalar_params(p1=None, p2=None, ring_radius=None, v_tang=None) -> dict:
    """Helper: pack the four nullable scalar columns."""
    return {
        "p1": p1,
        "p2": p2,
        "ring_radius": ring_radius,
        "initial_tangential_velocity": v_tang,
    }


def _row(time, particle_id, pos, vel, params, mass=None, charge=None) -> dict:
    """Build a single CSV row dict.

    `mass` and `charge` are optional — passing None leaves the column
    blank, matching how legacy worlds that don't track these properties
    leave them empty.
    """
    row = {
        "time": float(time),
        "particle_id": int(particle_id),
        "x": float(pos[0]),
        "y": float(pos[1]),
        "vx": float(vel[0]),
        "vy": float(vel[1]),
        **params,
    }
    if mass is not None:
        row["mass"] = float(mass)
    if charge is not None:
        row["charge"] = float(charge)
    return row


def _rows_two_particle(executor, exp_input, exp_output) -> list[dict]:
    """gravity / yukawa / fractional / diffusion / wave / oscillator / extra_dimensions."""
    p1 = float(exp_input["p1"])
    p2 = float(exp_input["p2"])
    pos2_init = list(exp_input["pos2"])
    vel2_init = list(exp_input["velocity2"])
    times = exp_output.get("measurement_times", [])
    pos1 = exp_output.get("pos1", [])
    pos2 = exp_output.get("pos2", [])
    vel1 = exp_output.get("velocity1", [])
    vel2 = exp_output.get("velocity2", [])
    params = _scalar_params(p1=p1, p2=p2)

    rows: list[dict] = []
    # t=0: particle 0 fixed at origin (agent-relative); particle 1 from input.
    rows.append(_row(0.0, 0, [0.0, 0.0], [0.0, 0.0], params))
    rows.append(_row(0.0, 1, pos2_init, vel2_init, params))
    # Observed times.
    for i, t in enumerate(times):
        if i < len(pos1) and i < len(vel1):
            rows.append(_row(t, 0, pos1[i], vel1[i], params))
        if i < len(pos2) and i < len(vel2):
            rows.append(_row(t, 1, pos2[i], vel2[i], params))
    return rows


def _rows_circle(executor, exp_input, exp_output) -> list[dict]:
    """11 particles: 0 = centre, 1-10 = ring."""
    ring_radius = float(exp_input.get("ring_radius", 5.0))
    v_tang = float(exp_input.get("initial_tangential_velocity", 0.0))
    times = exp_output.get("measurement_times", [])
    positions = exp_output.get("positions", [])
    velocities = exp_output.get("velocities", [])
    params = _scalar_params(ring_radius=ring_radius, v_tang=v_tang)

    n_ring = getattr(executor, "N_RING", 10)
    n_total = getattr(executor, "N_TOTAL", 11)
    angles = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    ring_pos = np.column_stack(
        [
            ring_radius * np.cos(angles),
            ring_radius * np.sin(angles),
        ]
    )
    ring_vel = np.column_stack(
        [
            -v_tang * np.sin(angles),
            v_tang * np.cos(angles),
        ]
    )
    init_positions = np.vstack([[[0.0, 0.0]], ring_pos])
    init_velocities = np.vstack([[[0.0, 0.0]], ring_vel])

    rows: list[dict] = []
    for pid in range(n_total):
        rows.append(_row(0.0, pid, init_positions[pid], init_velocities[pid], params))
    for i, t in enumerate(times):
        if i >= len(positions) or i >= len(velocities):
            break
        snap_p = positions[i]
        snap_v = velocities[i]
        for pid in range(min(n_total, len(snap_p), len(snap_v))):
            rows.append(_row(t, pid, snap_p[pid], snap_v[pid], params))
    return rows


def _rows_species(executor, exp_input, exp_output) -> list[dict]:
    """6 particles, agent supplies all positions and velocities."""
    init_positions = np.asarray(exp_input["positions"], dtype=float)
    init_velocities = np.asarray(exp_input["velocities"], dtype=float)
    times = exp_output.get("measurement_times", [])
    positions = exp_output.get("positions", [])
    velocities = exp_output.get("velocities", [])
    params = _scalar_params()

    n_total = getattr(executor, "N_PARTICLES", init_positions.shape[0])
    rows: list[dict] = []
    for pid in range(n_total):
        rows.append(_row(0.0, pid, init_positions[pid], init_velocities[pid], params))
    for i, t in enumerate(times):
        if i >= len(positions) or i >= len(velocities):
            break
        snap_p = positions[i]
        snap_v = velocities[i]
        for pid in range(min(n_total, len(snap_p), len(snap_v))):
            rows.append(_row(t, pid, snap_p[pid], snap_v[pid], params))
    return rows


def _rows_three_species(executor, exp_input, exp_output) -> list[dict]:
    """35 particles: 0-29 fixed background (zero initial velocity), 30-34 probes."""
    bg_positions = np.asarray(
        exp_output.get("background_initial_positions", []), dtype=float
    )
    if bg_positions.size == 0:
        bg_positions = np.asarray(executor._bg_positions_rel, dtype=float)
    n_bg = bg_positions.shape[0]
    bg_velocities = np.zeros_like(bg_positions)

    probe_pos = np.asarray(exp_input["probe_positions"], dtype=float)
    probe_vel = np.asarray(exp_input["probe_velocities"], dtype=float)
    init_positions = np.vstack([bg_positions, probe_pos])
    init_velocities = np.vstack([bg_velocities, probe_vel])

    n_total = init_positions.shape[0]
    times = exp_output.get("measurement_times", [])
    positions = exp_output.get("positions", [])
    velocities = exp_output.get("velocities", [])
    params = _scalar_params()

    rows: list[dict] = []
    for pid in range(n_total):
        rows.append(_row(0.0, pid, init_positions[pid], init_velocities[pid], params))
    for i, t in enumerate(times):
        if i >= len(positions) or i >= len(velocities):
            break
        snap_p = positions[i]
        snap_v = velocities[i]
        for pid in range(min(n_total, len(snap_p), len(snap_v))):
            rows.append(_row(t, pid, snap_p[pid], snap_v[pid], params))
    return rows


def _rows_dark_matter(executor, exp_input, exp_output) -> list[dict]:
    """25 agent-visible particles: 0-19 visible, 20-24 probes (dark hidden)."""
    visible_positions = np.asarray(
        exp_output.get("background_initial_positions", []), dtype=float
    )
    if visible_positions.size == 0:
        visible_positions = np.asarray(executor._visible_positions_rel, dtype=float)

    visible_velocities = np.asarray(
        getattr(executor, "_visible_velocities", np.zeros_like(visible_positions)),
        dtype=float,
    )
    if visible_velocities.shape != visible_positions.shape:
        visible_velocities = np.zeros_like(visible_positions)
    sign = float(exp_input.get("visible_velocity_sign", 1.0))
    visible_velocities = sign * visible_velocities

    probe_pos = np.asarray(exp_input["probe_positions"], dtype=float)
    probe_vel = np.asarray(exp_input["probe_velocities"], dtype=float)
    init_positions = np.vstack([visible_positions, probe_pos])
    init_velocities = np.vstack([visible_velocities, probe_vel])

    n_total = init_positions.shape[0]
    times = exp_output.get("measurement_times", [])
    positions = exp_output.get("positions", [])
    velocities = exp_output.get("velocities", [])
    params = _scalar_params()

    rows: list[dict] = []
    for pid in range(n_total):
        rows.append(_row(0.0, pid, init_positions[pid], init_velocities[pid], params))
    for i, t in enumerate(times):
        if i >= len(positions) or i >= len(velocities):
            break
        snap_p = positions[i]
        snap_v = velocities[i]
        for pid in range(min(n_total, len(snap_p), len(snap_v))):
            rows.append(_row(t, pid, snap_p[pid], snap_v[pid], params))
    return rows


def _rows_ether(executor, exp_input, exp_output) -> list[dict]:
    """26 particles: 0 anchor, 1-20 ring orbiters (fixed by world), 21-25 probes.

    Per-particle masses come from the executor output (`particle_masses`)
    so that probe masses set by the agent in this experiment are recorded
    on every row — the loss function can then reconstruct the agent-set
    initial state purely from the CSV.
    """
    bg_positions = np.asarray(
        exp_output.get("background_initial_positions", []), dtype=float
    )
    if bg_positions.size == 0:
        bg_positions = np.asarray(executor._bg_positions_rel, dtype=float)

    bg_velocities = np.asarray(
        exp_output.get("background_initial_velocities", []), dtype=float
    )
    if bg_velocities.shape != bg_positions.shape:
        bg_velocities = np.asarray(executor._bg_velocities, dtype=float)

    probe_pos = np.asarray(exp_input["probe_positions"], dtype=float)
    probe_vel = np.asarray(exp_input["probe_velocities"], dtype=float)
    init_positions = np.vstack([bg_positions, probe_pos])
    init_velocities = np.vstack([bg_velocities, probe_vel])

    masses = np.asarray(exp_output.get("particle_masses", []), dtype=float)
    n_total = init_positions.shape[0]
    if masses.shape != (n_total,):
        # Fallback: reconstruct from executor + agent-supplied probe_masses
        n_bg = executor._bg_masses.shape[0]
        n_probes = executor.N_PROBES
        probe_masses = np.asarray(
            exp_input.get(
                "probe_masses",
                [executor.DEFAULT_PROBE_MASS] * n_probes,
            ),
            dtype=float,
        )
        masses = np.concatenate([executor._bg_masses, probe_masses])

    times = exp_output.get("measurement_times", [])
    positions = exp_output.get("positions", [])
    velocities = exp_output.get("velocities", [])
    params = _scalar_params()

    rows: list[dict] = []
    for pid in range(n_total):
        rows.append(
            _row(
                0.0,
                pid,
                init_positions[pid],
                init_velocities[pid],
                params,
                mass=float(masses[pid]),
            )
        )
    for i, t in enumerate(times):
        if i >= len(positions) or i >= len(velocities):
            break
        snap_p = positions[i]
        snap_v = velocities[i]
        for pid in range(min(n_total, len(snap_p), len(snap_v))):
            rows.append(
                _row(
                    t,
                    pid,
                    snap_p[pid],
                    snap_v[pid],
                    params,
                    mass=float(masses[pid]),
                )
            )
    return rows


# The Hubble world has the same on-CSV shape as ether (26 particles, masses
# in the per-row mass column, agent-controllable probes); the row builder is
# identical. We alias ``_rows_hubble = _rows_ether`` to make the registry
# explicit without duplicating code.
_rows_hubble = _rows_ether


def _rows_coulomb_easy(executor, exp_input, exp_output) -> list[dict]:
    """2 particles: 0 = pinned source, 1 = mobile probe.

    Mirrors ``_rows_two_particle`` (p1, p2 stored on every row) but also
    fills the per-particle ``charge`` column with the *signed* charges the
    executor used internally so the loader can reconstruct the dynamics.
    """
    p1 = float(exp_input["p1"])
    p2 = float(exp_input["p2"])
    pos2_init = list(exp_input["pos2"])
    vel2_init = list(exp_input["velocity2"])
    times = exp_output.get("measurement_times", [])
    pos1 = exp_output.get("pos1", [])
    pos2 = exp_output.get("pos2", [])
    vel1 = exp_output.get("velocity1", [])
    vel2 = exp_output.get("velocity2", [])
    params = _scalar_params(p1=p1, p2=p2)

    # Executor enforces opposite-sign charges to guarantee attraction.
    charges = exp_output.get("particle_charges", [+abs(p1), -abs(p2)])

    rows: list[dict] = []
    rows.append(_row(0.0, 0, [0.0, 0.0], [0.0, 0.0], params, charge=charges[0]))
    rows.append(_row(0.0, 1, pos2_init, vel2_init, params, charge=charges[1]))
    for i, t in enumerate(times):
        if i < len(pos1) and i < len(vel1):
            rows.append(_row(t, 0, pos1[i], vel1[i], params, charge=charges[0]))
        if i < len(pos2) and i < len(vel2):
            rows.append(_row(t, 1, pos2[i], vel2[i], params, charge=charges[1]))
    return rows


def _rows_coulomb_hard(executor, exp_input, exp_output) -> list[dict]:
    """10 particles, all mobile, signed per-particle charges from the agent."""
    init_positions = np.asarray(exp_input["positions"], dtype=float)
    init_velocities = np.asarray(exp_input["velocities"], dtype=float)
    charges_in = np.asarray(exp_input["charges"], dtype=float)
    times = exp_output.get("measurement_times", [])
    positions = exp_output.get("positions", [])
    velocities = exp_output.get("velocities", [])
    params = _scalar_params()

    n_total = getattr(executor, "N_PARTICLES", init_positions.shape[0])

    rows: list[dict] = []
    for pid in range(n_total):
        rows.append(
            _row(
                0.0,
                pid,
                init_positions[pid],
                init_velocities[pid],
                params,
                charge=float(charges_in[pid]),
            )
        )
    for i, t in enumerate(times):
        if i >= len(positions) or i >= len(velocities):
            break
        snap_p = positions[i]
        snap_v = velocities[i]
        for pid in range(min(n_total, len(snap_p), len(snap_v))):
            rows.append(
                _row(
                    t,
                    pid,
                    snap_p[pid],
                    snap_v[pid],
                    params,
                    charge=float(charges_in[pid]),
                )
            )
    return rows


_RowBuilders = {
    "gravity": _rows_two_particle,
    "yukawa": _rows_two_particle,
    "fractional": _rows_two_particle,
    "diffusion": _rows_two_particle,
    "wave": _rows_two_particle,
    # ``oscillator`` shares the 2-particle (p1, p2, pos2, velocity2,
    # measurement_times) protocol; the trajectory log doesn't currently
    # carry the optional ``start_time``, but the per-time particle rows
    # are correct because the executor returns the standard pos1/pos2
    # arrays measured at the requested local times.
    "oscillator": _rows_two_particle,
    "circle": _rows_circle,
    "species": _rows_species,
    "three_species": _rows_three_species,
    "dark_matter": _rows_dark_matter,
    "ether": _rows_ether,
    "hubble": _rows_hubble,
    "coulomb_easy": _rows_coulomb_easy,
    "coulomb_hard": _rows_coulomb_hard,
    "extra_dimensions": _rows_two_particle,
}
