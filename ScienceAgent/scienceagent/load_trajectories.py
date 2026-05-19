"""
Load trajectory CSVs into ready-to-fit numpy arrays.

A `TrajectoryLogger` writes one row per (experiment, time, particle) to
`results/trajectories/<world>.csv`. This module reads that CSV back and
returns a list of `ExperimentTrajectory` dataclass instances, each with
sorted-by-time arrays of shape (T, N, 2) for positions and velocities,
plus the t=0 slice exposed separately for convenience.

Typical use
-----------
    from scienceagent.load_trajectories import load_trajectories

    for exp in load_trajectories("gravity"):
        pred = proposed_law(
            exp.initial_positions,
            exp.initial_velocities,
            exp.times,
            **exp.params,
        )
        mse = ((pred - exp.positions) ** 2).mean()

CLI
---
    python -m scienceagent.load_trajectories gravity
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

# Scalar params that may appear as columns. Only the populated ones are
# attached to each experiment's `params` dict.
_PARAM_COLUMNS: tuple[str, ...] = (
    "p1",
    "p2",
    "ring_radius",
    "initial_tangential_velocity",
)


@dataclass
class ExperimentTrajectory:
    """One experiment's trajectory in fit-ready numpy form.

    Shapes
    ------
    times:              (T,)        sorted, includes 0.0 (the t=0 row)
    positions:          (T, N, 2)
    velocities:         (T, N, 2)
    initial_positions:  (N, 2)      == positions[0]
    initial_velocities: (N, 2)      == velocities[0]
    masses:             (N,)        per-particle masses; NaN-filled for
                                    legacy worlds that don't track mass
    charges:            (N,)        per-particle charges; NaN-filled for
                                    worlds that don't track charge
    """

    run_id: str
    experiment_id: str
    round: int
    source: str
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    initial_positions: np.ndarray
    initial_velocities: np.ndarray
    masses: np.ndarray
    charges: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.times.shape[0])


def default_csv_path(world: str) -> Path:
    """Return the canonical CSV path for `world` under the repo's results dir."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "results" / "trajectories" / f"{world}.csv"


def load_trajectories(
    world_or_path: str | os.PathLike,
    *,
    run_id: Optional[str | Sequence[str]] = None,
) -> list[ExperimentTrajectory]:
    """Load all experiments from a per-world trajectory CSV.

    Args:
        world_or_path: Either a world name (e.g. "gravity") which is
            resolved to `results/trajectories/<world>.csv`, or an explicit
            path to a CSV file.
        run_id: Optional filter. Pass a single string or a collection of
            strings to keep only experiments whose `run_id` matches.

    Returns:
        A list of `ExperimentTrajectory`, one per `experiment_id`, in the
        order the experiments appear in the file. Experiments whose stored
        positions/velocities contain NaN values (typically left over from
        an earlier buggy simulator run) are skipped with a warning; this
        prevents one corrupted experiment from poisoning the whole load.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if a single experiment has a truly ragged
            (time, particle) grid — i.e. some (time, particle_id) cell was
            never written at all (distinct from "written with NaN values").
    """
    path = _resolve_path(world_or_path)
    run_id_filter = _normalise_run_id(run_id)

    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if run_id_filter is not None and row["run_id"] not in run_id_filter:
                continue
            grouped.setdefault(row["experiment_id"], []).append(row)

    out: list[ExperimentTrajectory] = []
    for eid, rows in grouped.items():
        exp = _build_experiment(eid, rows)
        if exp is None:
            continue
        out.append(exp)
    return out


# ---------------------------------------------------------------------------
# Internals.


def _resolve_path(world_or_path: str | os.PathLike) -> Path:
    path = Path(world_or_path)
    if path.suffix == ".csv" or path.exists():
        if not path.exists():
            raise FileNotFoundError(f"Trajectory CSV not found: {path}")
        return path
    candidate = default_csv_path(str(world_or_path))
    if not candidate.exists():
        raise FileNotFoundError(
            f"No trajectory CSV for world '{world_or_path}'. "
            f"Expected at {candidate}."
        )
    return candidate


def _normalise_run_id(
    run_id: Optional[str | Sequence[str]],
) -> Optional[set[str]]:
    if run_id is None:
        return None
    if isinstance(run_id, str):
        return {run_id}
    return set(run_id)


def _build_experiment(
    experiment_id: str, rows: list[dict]
) -> Optional[ExperimentTrajectory]:
    """Reshape the rows of one experiment into (T, N, 2) arrays.

    Returns None (with a warning) if the experiment's rows contain NaN
    values for positions/velocities — typically a leftover from an earlier
    buggy simulator run. Truly missing rows (cells that were never
    written) still raise ValueError, since that signals a logging bug
    rather than data quality.
    """
    if not rows:
        raise ValueError(f"Empty row list for experiment '{experiment_id}'.")

    # Unique sorted times and particles.
    times = sorted({float(r["time"]) for r in rows})
    particles = sorted({int(r["particle_id"]) for r in rows})

    # Index lookups for O(1) fill below.
    time_index = {t: i for i, t in enumerate(times)}
    particle_index = {pid: j for j, pid in enumerate(particles)}

    T = len(times)
    N = len(particles)
    positions = np.full((T, N, 2), np.nan, dtype=np.float64)
    velocities = np.full((T, N, 2), np.nan, dtype=np.float64)
    # Track which (time, particle) cells we actually saw a row for, so we
    # can distinguish "row never written" from "row written with NaN".
    written = np.zeros((T, N), dtype=bool)
    # Mass / charge columns are per-particle (constant across time) but
    # may be empty for worlds that never wrote them. We collect per-
    # particle entries and leave NaN where the column was blank.
    masses = np.full((N,), np.nan, dtype=np.float64)
    charges = np.full((N,), np.nan, dtype=np.float64)

    for r in rows:
        i = time_index[float(r["time"])]
        j = particle_index[int(r["particle_id"])]
        positions[i, j] = (float(r["x"]), float(r["y"]))
        velocities[i, j] = (float(r["vx"]), float(r["vy"]))
        written[i, j] = True
        m_raw = r.get("mass")
        if m_raw not in (None, "", "NaN", "nan"):
            masses[j] = float(m_raw)
        q_raw = r.get("charge")
        if q_raw not in (None, "", "NaN", "nan"):
            charges[j] = float(q_raw)

    if not written.all():
        missing_idx = np.argwhere(~written)
        first = tuple(missing_idx[0])
        raise ValueError(
            f"Experiment '{experiment_id}' has a ragged (time, particle) grid; "
            f"first missing cell at (time_index, particle_index)={first}."
        )

    if np.isnan(positions).any() or np.isnan(velocities).any():
        # Rows were present but contained NaN values — typically corrupt
        # data from an earlier buggy run. Skip rather than crash so a
        # later good experiment in the same CSV still loads.
        nan_cells = int(np.isnan(positions[..., 0]).sum())
        import warnings as _warnings

        _warnings.warn(
            f"skipping experiment '{experiment_id}': "
            f"{nan_cells}/{T*N} (time, particle) cells contain NaN values "
            f"in positions/velocities (likely stale rows from a prior buggy run).",
            stacklevel=3,
        )
        return None

    # All these run-level fields must be identical across rows of the same
    # experiment, so we read them off the first row.
    head = rows[0]
    params = {
        col: float(head[col])
        for col in _PARAM_COLUMNS
        if head.get(col) not in (None, "", "NaN", "nan")
    }

    times_arr = np.asarray(times, dtype=np.float64)
    return ExperimentTrajectory(
        run_id=head["run_id"],
        experiment_id=experiment_id,
        round=int(head["round"]),
        source=head["source"],
        times=times_arr,
        positions=positions,
        velocities=velocities,
        initial_positions=positions[0].copy(),
        initial_velocities=velocities[0].copy(),
        masses=masses,
        charges=charges,
        params=params,
    )


# ---------------------------------------------------------------------------
# CLI: `python -m scienceagent.load_trajectories <world> [--run-id ...]`


def _summarise(experiments: list[ExperimentTrajectory]) -> str:
    if not experiments:
        return "(no experiments loaded)"
    by_run: "OrderedDict[str, list[ExperimentTrajectory]]" = OrderedDict()
    for e in experiments:
        by_run.setdefault(e.run_id, []).append(e)

    lines = [f"loaded {len(experiments)} experiments across {len(by_run)} run(s)"]
    for run_id, exps in by_run.items():
        rounds = sorted({e.round for e in exps})
        sources = sorted({e.source for e in exps})
        Ns = sorted({e.n_particles for e in exps})
        Ts = sorted({e.n_times for e in exps})
        lines.append(
            f"  {run_id}"
            f"  experiments={len(exps)}"
            f"  rounds={rounds}"
            f"  source={','.join(sources)}"
            f"  N={Ns}"
            f"  T={Ts}"
        )
    first = experiments[0]
    lines.append(
        f"first experiment: id={first.experiment_id} "
        f"shape positions={first.positions.shape} "
        f"params={first.params}"
    )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load and summarise a trajectory CSV.",
    )
    parser.add_argument(
        "world",
        help="World name (resolves to results/trajectories/<world>.csv) "
        "or an explicit path to a CSV file.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="Filter to one or more run_ids (repeatable).",
    )
    args = parser.parse_args(argv)

    try:
        experiments = load_trajectories(args.world, run_id=args.run_id)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(_summarise(experiments))
    return 0


if __name__ == "__main__":
    sys.exit(main())
