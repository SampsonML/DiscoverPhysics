"""
Run an MSE fit of an agent-proposed law against this run's trajectory CSV.

The fit reuses the same scipy.optimize machinery as the post-discovery
evaluator (`_maybe_fit`, `_two_particle_loss`, `_circle_loss`) so the
mid-run feedback the agent gets matches the eventual evaluation metric.

Public entry point
------------------
    fit_law(law_source, world, csv_path, run_id) -> dict

Returns a JSON-serialisable result dict with keys:

    loss_before        | float | None  (MSE at the agent's init params)
    loss_after         | float | None  (MSE after scipy.optimize)
    fitted_params      | dict[str, float]
    declared_params    | dict[str, dict]   (init + bounds the agent gave)
    n_training         | int               (training trajectories used)
    error              | str | None        (compile / fit / optimizer error)

CLI
---
    python -m scienceagent.mse_fitting <world> <run_id> <law_source_file>
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from scienceagent.evaluator import (
    _compile_fit_parameters,
    _compile_law,
    _coulomb_hard_loss,
    _dark_matter_loss,
    _ether_loss,
    _fit_law_parameters,
    _three_species_loss,
    _two_particle_loss,
    _circle_loss,
    _validate_fit_spec,
    clean_law_source,
)
from scienceagent.load_trajectories import (
    ExperimentTrajectory,
    default_csv_path,
    load_trajectories,
)

# Worlds that map to a position-MSE loss function. Continuous-parameter
# worlds (gravity, yukawa, fractional, diffusion, wave, circle) match the
# evaluator's existing fitting set; structural worlds (three_species,
# dark_matter) score against the trajectory data even when no parameters
# are declared, so the agent can see how well its current law tracks.
_LOSS_FNS: dict[str, Callable] = {
    "gravity": _two_particle_loss,
    "yukawa": _two_particle_loss,
    "fractional": _two_particle_loss,
    "diffusion": _two_particle_loss,
    "wave": _two_particle_loss,
    # ``oscillator`` shares the 2-particle agent-facing law signature
    # ``discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params)``,
    # so the same trajectory MSE loss applies even though the hidden
    # physics is time-varying.  The agent typically declares period /
    # amplitude / phase as fittable parameters.
    "oscillator": _two_particle_loss,
    "extra_dimensions": _two_particle_loss,
    "circle": _circle_loss,
    "three_species": _three_species_loss,
    "dark_matter": _dark_matter_loss,
    "ether": _ether_loss,
    # Hubble has the same agent-facing law shape as ether (26 particles,
    # masses argument, scoring on the probe slice 21:26), so the same
    # loss function applies.
    "hubble": _ether_loss,
    # coulomb_easy uses the same 2-particle protocol as gravity/yukawa/...
    # (law signature: pos1, pos2, p1, p2, velocity2, duration).
    "coulomb_easy": _two_particle_loss,
    "coulomb_hard": _coulomb_hard_loss,
}


def redact_world_leak(
    msg: str, csv_path: Optional[os.PathLike | str], world: str
) -> str:
    """Strip world-name and trajectory-CSV path substrings from `msg`.

    Why: error strings are echoed back to the agent inside <mse_fit_output>;
    the CSV is named results/trajectories/<world>.csv, so any raw exception
    message that quotes the path (e.g. pandas FileNotFoundError) hands the
    agent the answer it's trying to discover.
    """
    s = str(msg)
    needles: list[str] = []
    if csv_path is not None:
        p = Path(csv_path)
        needles.extend([str(p), str(p.resolve()), p.name])
    if world:
        needles.append(world)
    for needle in needles:
        if needle:
            s = s.replace(needle, "<redacted>")
    return s


def fit_law(
    law_source: str,
    world: str,
    csv_path: Optional[os.PathLike | str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Compile and fit `law_source` against this run's CSV trajectories.

    Args:
        law_source: Python source string defining `discovered_law` and
            (optionally) `fit_parameters`. Same format the agent uses in
            <final_law>.
        world: World name (used to pick the loss function and default CSV).
        csv_path: Override path to the trajectory CSV. Defaults to
            `results/trajectories/<world>.csv`.
        run_id: Filter the CSV to this run only. Required.

    Returns:
        Result dict (see module docstring). Always returns; errors are
        reported via the `error` field rather than raising.
    """
    result = {
        "loss_before": None,
        "loss_after": None,
        "fitted_params": {},
        "declared_params": {},
        "n_training": 0,
        "error": None,
    }

    loss_fn = _LOSS_FNS.get(world)
    if loss_fn is None:
        result["error"] = (
            f"world '{world}' has no continuous parameters to fit; "
            f"supported worlds: {sorted(_LOSS_FNS)}"
        )
        return result

    if not run_id:
        result["error"] = "run_id is required so the fit only sees this run's data"
        return result

    csv_path = Path(csv_path) if csv_path else default_csv_path(world)
    if not csv_path.exists():
        result["error"] = (
            "trajectory CSV not found for this run; run at least one "
            "successful <run_experiment> first"
        )
        return result

    try:
        experiments = load_trajectories(csv_path, run_id=run_id)
    except Exception as e:
        result["error"] = (
            f"failed to load training data: {redact_world_leak(e, csv_path, world)}"
        )
        return result
    if not experiments:
        result["error"] = f"no training trajectories logged yet for run_id={run_id}"
        return result

    try:
        training = _experiments_to_training(world, experiments)
    except Exception as e:
        result["error"] = (
            "failed to reshape CSV into training samples: "
            + redact_world_leak(e, csv_path, world)
        )
        return result
    result["n_training"] = len(training)
    if not training:
        result["error"] = "no usable training samples after CSV reshape"
        return result

    try:
        discovered_law = _compile_law(law_source)
    except Exception as e:
        result["error"] = "compile_error: " + redact_world_leak(e, csv_path, world)
        return result

    fit_fn = _compile_fit_parameters(law_source)
    fit_spec_list: list = []
    if fit_fn is not None:
        try:
            raw_spec = fit_fn()
            fit_spec_list = _validate_fit_spec(raw_spec)
        except Exception as e:
            result["error"] = "invalid_fit_parameters: " + redact_world_leak(
                e, csv_path, world
            )
            return result
        result["declared_params"] = {
            name: {"init": init, "bounds": list(bounds)}
            for name, init, bounds in fit_spec_list
        }

    init_kwargs = {name: init for name, init, _ in fit_spec_list}
    init_law = (
        functools.partial(discovered_law, **init_kwargs)
        if init_kwargs
        else discovered_law
    )
    try:
        loss_before = float(loss_fn(init_law, training))
    except Exception:
        loss_before = float("inf")
    result["loss_before"] = _finite_or_none(loss_before)

    if not fit_spec_list:
        # No fit_parameters() — report the law's MSE as-is.
        result["loss_after"] = result["loss_before"]
        result["fitted_params"] = {}
        return result

    try:
        fitted = _fit_law_parameters(discovered_law, fit_spec_list, training, loss_fn)
    except Exception as e:
        result["error"] = "optimizer_failure: " + redact_world_leak(e, csv_path, world)
        result["fitted_params"] = init_kwargs
        result["loss_after"] = result["loss_before"]
        return result

    bound_law = functools.partial(discovered_law, **fitted)
    try:
        loss_after = float(loss_fn(bound_law, training))
    except Exception:
        loss_after = float("inf")

    result["fitted_params"] = fitted
    result["loss_after"] = _finite_or_none(loss_after)
    return result


def _finite_or_none(x: float) -> Optional[float]:
    return None if not np.isfinite(x) else float(x)


# Per-world ExperimentTrajectory → training-sample packers, registrable
# without editing this module. Mirrors the ``_LOSS_FNS`` extension point:
# ``mse_fitting_private`` populates this for held-out worlds so their
# names never appear in the public source. Takes precedence over the
# built-in routing below.
_PACK_FNS: dict[str, Callable[[ExperimentTrajectory], dict]] = {}


def _experiments_to_training(
    world: str,
    experiments: list[ExperimentTrajectory],
) -> list[dict]:
    """Convert ExperimentTrajectory list into the {"input", "output"} shape
    the loss functions in evaluator.py expect."""
    pack = _PACK_FNS.get(world)
    if pack is not None:
        return [pack(e) for e in experiments]
    if world == "circle":
        return [_one_circle_sample(e) for e in experiments]
    if world in ("ether", "hubble"):
        # Same on-CSV layout — 26 particles + per-particle masses.
        return [_one_ether_sample(e) for e in experiments]
    if world in ("three_species", "dark_matter"):
        # Both worlds use the same shape: full N-particle initial state
        # from the CSV's t=0 row + observed positions at t > 0. Loss
        # functions decide which particles to score.
        return [_one_full_state_sample(e) for e in experiments]
    if world == "coulomb_hard":
        return [_one_coulomb_hard_sample(e) for e in experiments]
    return [_one_two_particle_sample(e) for e in experiments]


def _one_two_particle_sample(exp: ExperimentTrajectory) -> dict:
    """Pack a 2-particle ExperimentTrajectory into evaluator-loss format.

    The loss function reads pos2/measurement_times from `output` and
    p1/p2/pos2/velocity2 from `input`. Only t > 0 rows are observations;
    t = 0 carries the initial conditions.
    """
    times_obs = exp.times[1:].tolist()  # (T-1,)
    pos2_obs = exp.positions[1:, 1, :].tolist()  # particle index 1 = mobile probe
    vel2_obs = exp.velocities[1:, 1, :].tolist()
    pos2_init = exp.initial_positions[1].tolist()
    vel2_init = exp.initial_velocities[1].tolist()
    return {
        "input": {
            "p1": exp.params.get("p1"),
            "p2": exp.params.get("p2"),
            "pos2": pos2_init,
            "velocity2": vel2_init,
            "measurement_times": times_obs,
        },
        "output": {
            "measurement_times": times_obs,
            "pos2": pos2_obs,
            "velocity2": vel2_obs,
        },
    }


def _one_circle_sample(exp: ExperimentTrajectory) -> dict:
    """Pack a circle-world ExperimentTrajectory into evaluator-loss format."""
    times_obs = exp.times[1:].tolist()
    positions_obs = exp.positions[1:].tolist()  # (T-1, 11, 2)
    velocities_obs = exp.velocities[1:].tolist()
    return {
        "input": {
            "ring_radius": exp.params.get("ring_radius"),
            "initial_tangential_velocity": exp.params.get(
                "initial_tangential_velocity"
            ),
            "measurement_times": times_obs,
        },
        "output": {
            "measurement_times": times_obs,
            "positions": positions_obs,
            "velocities": velocities_obs,
        },
    }


def _one_full_state_sample(exp: ExperimentTrajectory) -> dict:
    """Pack a multi-particle ExperimentTrajectory (three_species,
    dark_matter, ...) into evaluator-loss format.

    The full N-particle initial state goes into `input.init_positions /
    init_velocities` (read directly from the CSV t=0 row, so this
    correctly carries the visible-particle initial velocities the agent
    cannot observe directly in the dark-matter world). Observed positions
    at t > 0 go into `output.positions`.
    """
    times_obs = exp.times[1:].tolist()
    positions_obs = exp.positions[1:].tolist()  # (T-1, N, 2)
    return {
        "input": {
            "init_positions": exp.initial_positions.tolist(),  # (N, 2)
            "init_velocities": exp.initial_velocities.tolist(),  # (N, 2)
            "measurement_times": times_obs,
        },
        "output": {
            "measurement_times": times_obs,
            "positions": positions_obs,
        },
    }


def _one_coulomb_hard_sample(exp: ExperimentTrajectory) -> dict:
    """Pack a coulomb_hard ExperimentTrajectory into evaluator-loss format.

    Same shape as ``_one_full_state_sample`` but additionally carries the
    per-particle signed charges recorded in the CSV ``charge`` column —
    the coulomb_hard ``discovered_law`` takes ``charges`` as a third
    argument, and ``_coulomb_hard_loss`` reads them from
    ``input["init_charges"]``.
    """
    times_obs = exp.times[1:].tolist()
    positions_obs = exp.positions[1:].tolist()  # (T-1, 10, 2)
    charges = exp.charges
    if charges is None or np.isnan(charges).any():
        # The row builder always writes the charge column for coulomb_hard,
        # so a NaN here means a partially-populated CSV. Default to zeros
        # so the fit doesn't crash, even though loss will be uninformative.
        charges = np.zeros(exp.n_particles, dtype=float)
    return {
        "input": {
            "init_positions": exp.initial_positions.tolist(),  # (10, 2)
            "init_velocities": exp.initial_velocities.tolist(),  # (10, 2)
            "init_charges": charges.tolist(),  # (10,)
            "measurement_times": times_obs,
        },
        "output": {
            "measurement_times": times_obs,
            "positions": positions_obs,
        },
    }


def _one_ether_sample(exp: ExperimentTrajectory) -> dict:
    """Pack an ether-world ExperimentTrajectory into evaluator-loss format.

    Same shape as ``_one_full_state_sample`` but additionally carries the
    per-particle masses recorded in the CSV ``mass`` column — the ether
    ``discovered_law`` takes ``masses`` as a third argument, and
    ``_ether_loss`` reads them from ``input["init_masses"]``.
    """
    times_obs = exp.times[1:].tolist()
    positions_obs = exp.positions[1:].tolist()  # (T-1, 26, 2)
    masses = exp.masses
    if masses is None or np.isnan(masses).any():
        # Should not happen for ether (the row builder always writes the
        # mass column) but fall back gracefully so a partial CSV doesn't
        # crash the fit.
        masses = np.ones(exp.n_particles, dtype=float)
    return {
        "input": {
            "init_positions": exp.initial_positions.tolist(),  # (26, 2)
            "init_velocities": exp.initial_velocities.tolist(),  # (26, 2)
            "init_masses": masses.tolist(),  # (26,)
            "measurement_times": times_obs,
        },
        "output": {
            "measurement_times": times_obs,
            "positions": positions_obs,
        },
    }


# ---------------------------------------------------------------------------
# CLI: `python -m scienceagent.mse_fitting <world> <run_id> <law_file>`


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit a candidate law against a run's trajectory CSV."
    )
    parser.add_argument("world", help="World name (e.g. gravity, yukawa, circle).")
    parser.add_argument("run_id", help="run_id to filter the CSV by.")
    parser.add_argument(
        "law_file",
        help="Path to a Python file defining discovered_law (and optionally fit_parameters).",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Override the trajectory CSV path "
        "(default: results/trajectories/<world>.csv).",
    )
    args = parser.parse_args(argv)

    try:
        law_source = Path(args.law_file).read_text()
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    result = fit_law(
        law_source=law_source,
        world=args.world,
        csv_path=args.csv_path,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, default=float))
    return 0 if result["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
