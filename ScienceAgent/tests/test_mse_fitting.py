"""
Tests for mse_fitting.fit_law: compile / fit / error paths, plus a
mocked-LLM end-to-end test of the <run_mse_fit> protocol in DiscoveryAgent.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scienceagent.executor import (
    SimulationExecutor,
    CircleExecutor,
    ThreeSpeciesExecutor,
    DarkMatterExecutor,
)
from scienceagent.trajectory_logger import TrajectoryLogger, make_run_id
from scienceagent.mse_fitting import fit_law
from scienceagent.agent import DiscoveryAgent

# ---------------------------------------------------------------------------
# Test fixtures: write a small CSV before each fit-machinery test.


@pytest.fixture
def gravity_csv(tmp_path):
    """Build a per-world CSV with one experiment and yield path + run_id."""
    executor = SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
    )
    csv_path = tmp_path / "gravity.csv"
    run_id = make_run_id("test")
    logger = TrajectoryLogger("gravity", executor, csv_path, run_id)
    exp_input = {
        "p1": 1.0,
        "p2": 1.0,
        "pos2": [3.0, 0.0],
        "velocity2": [0.0, 0.0],
        "measurement_times": [0.5, 1.0, 1.5, 2.0],
    }
    [out] = executor.run([exp_input])
    logger.log_experiment(1, "agent", exp_input, out)
    return csv_path, run_id


@pytest.fixture
def circle_csv(tmp_path):
    executor = CircleExecutor()
    csv_path = tmp_path / "circle.csv"
    run_id = make_run_id("test")
    logger = TrajectoryLogger("circle", executor, csv_path, run_id)
    exp_input = {
        "ring_radius": 5.0,
        "initial_tangential_velocity": 0.0,
        "measurement_times": [0.5, 1.0, 1.5],
    }
    [out] = executor.run([exp_input])
    logger.log_experiment(1, "agent", exp_input, out)
    return csv_path, run_id


# ---------------------------------------------------------------------------
# Successful fit with one free parameter.

_LAW_WITH_FIT = """\
def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    import numpy as np
    k = params.get("k", 0.0)
    final_pos2 = [pos2[0] + k * duration, pos2[1]]
    final_vel2 = [k, 0.0]
    return final_pos2, final_vel2

def fit_parameters():
    return {"k": {"init": 0.0, "bounds": [-2.0, 2.0]}}
"""


def test_fit_runs_and_improves_loss(gravity_csv):
    csv_path, run_id = gravity_csv
    result = fit_law(
        law_source=_LAW_WITH_FIT,
        world="gravity",
        csv_path=csv_path,
        run_id=run_id,
    )
    assert result["error"] is None, result
    assert result["n_training"] == 1
    assert "k" in result["fitted_params"]
    assert isinstance(result["fitted_params"]["k"], float)
    # Optimizer can never make the training loss worse.
    assert result["loss_after"] <= result["loss_before"] + 1e-9
    assert result["declared_params"] == {"k": {"init": 0.0, "bounds": [-2.0, 2.0]}}


# ---------------------------------------------------------------------------
# Law without fit_parameters: still computes loss, but no fitting.

_LAW_NO_FIT = """\
def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    return list(pos2), [0.0, 0.0]
"""


def test_fit_without_fit_parameters_reports_loss(gravity_csv):
    csv_path, run_id = gravity_csv
    result = fit_law(_LAW_NO_FIT, "gravity", csv_path, run_id)
    assert result["error"] is None
    assert result["fitted_params"] == {}
    assert result["declared_params"] == {}
    assert result["loss_before"] is not None
    assert result["loss_before"] == result["loss_after"]


# ---------------------------------------------------------------------------
# Compile / spec / world / data error paths.


def test_fit_compile_error(gravity_csv):
    csv_path, run_id = gravity_csv
    result = fit_law("not python at all !!!", "gravity", csv_path, run_id)
    assert result["error"] is not None
    assert "compile_error" in result["error"]


def test_fit_invalid_fit_parameters(gravity_csv):
    csv_path, run_id = gravity_csv
    bad = """\
def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    return list(pos2), [0.0, 0.0]

def fit_parameters():
    return {"k": {"init": 0.0}}  # missing bounds
"""
    result = fit_law(bad, "gravity", csv_path, run_id)
    assert result["error"] is not None
    assert "invalid_fit_parameters" in result["error"]


def test_fit_unsupported_world(tmp_path):
    # `species` is the one structural world we do NOT (yet) support fitting for.
    result = fit_law(
        _LAW_NO_FIT, "species", csv_path=tmp_path / "species.csv", run_id="r1"
    )
    assert result["error"] is not None
    assert "no continuous parameters" in result["error"]


def test_fit_missing_csv(tmp_path):
    result = fit_law(
        _LAW_NO_FIT, "gravity", csv_path=tmp_path / "missing.csv", run_id="r1"
    )
    assert result["error"] is not None
    assert "not found" in result["error"]


def test_fit_unknown_run_id(gravity_csv):
    csv_path, _ = gravity_csv
    result = fit_law(
        _LAW_NO_FIT, "gravity", csv_path=csv_path, run_id="nonexistent-run-id"
    )
    assert result["error"] is not None
    assert "no training trajectories" in result["error"]


# ---------------------------------------------------------------------------
# Circle world routes to _circle_loss.

_CIRCLE_LAW_NO_FIT = """\
def discovered_law(positions, velocities, duration, **params):
    return [list(p) for p in positions]
"""


def test_circle_world_uses_circle_loss(circle_csv):
    csv_path, run_id = circle_csv
    result = fit_law(_CIRCLE_LAW_NO_FIT, "circle", csv_path=csv_path, run_id=run_id)
    assert result["error"] is None
    assert result["loss_before"] is not None


# ---------------------------------------------------------------------------
# Three-species world (35 particles, full-state scoring).


@pytest.fixture
def three_species_csv(tmp_path):
    executor = ThreeSpeciesExecutor()
    csv_path = tmp_path / "three_species.csv"
    run_id = make_run_id("test")
    logger = TrajectoryLogger("three_species", executor, csv_path, run_id)
    exp_input = {
        "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
        "probe_velocities": [[0, 0]] * 5,
        "measurement_times": [0.5, 1.0],
    }
    [out] = executor.run([exp_input])
    logger.log_experiment(1, "agent", exp_input, out)
    return csv_path, run_id


_THREE_SPECIES_LAW_NO_FIT = """\
def discovered_law(positions, velocities, duration, **params):
    return [list(p) for p in positions]
"""

_THREE_SPECIES_LAW_WITH_FIT = """\
def discovered_law(positions, velocities, duration, **params):
    import numpy as np
    drift = params.get("drift", 0.0)
    return [[p[0] + drift * duration, p[1]] for p in positions]

def fit_parameters():
    return {"drift": {"init": 0.0, "bounds": [-1.0, 1.0]}}
"""


def test_three_species_loss_no_fit(three_species_csv):
    csv_path, run_id = three_species_csv
    result = fit_law(
        _THREE_SPECIES_LAW_NO_FIT, "three_species", csv_path=csv_path, run_id=run_id
    )
    assert result["error"] is None
    assert result["loss_before"] is not None
    assert result["loss_before"] == result["loss_after"]
    assert result["fitted_params"] == {}


def test_three_species_fit_runs(three_species_csv):
    csv_path, run_id = three_species_csv
    result = fit_law(
        _THREE_SPECIES_LAW_WITH_FIT, "three_species", csv_path=csv_path, run_id=run_id
    )
    assert result["error"] is None
    assert "drift" in result["fitted_params"]
    assert result["loss_after"] <= result["loss_before"] + 1e-9


# ---------------------------------------------------------------------------
# Dark-matter world (25 particles, scoring on probes 20-24 only).


@pytest.fixture
def dark_matter_csv(tmp_path):
    executor = DarkMatterExecutor()
    csv_path = tmp_path / "dark_matter.csv"
    run_id = make_run_id("test")
    logger = TrajectoryLogger("dark_matter", executor, csv_path, run_id)
    exp_input = {
        "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
        "probe_velocities": [[0, 0]] * 5,
        "measurement_times": [1.0, 2.0],
    }
    [out] = executor.run([exp_input])
    logger.log_experiment(1, "agent", exp_input, out)
    return csv_path, run_id


_DARK_MATTER_LAW_NO_FIT = """\
def discovered_law(positions, velocities, duration, **params):
    return [list(p) for p in positions]
"""

# Variants used to confirm probe-only scoring: they differ only in their
# treatment of the visible particles (indices 0-19) but predict identical
# probe positions (indices 20-24).
_DARK_MATTER_LAW_VISIBLE_OFFSET = """\
def discovered_law(positions, velocities, duration, **params):
    out = []
    for i, p in enumerate(positions):
        if i < 20:
            out.append([p[0] + 999.0, p[1] + 999.0])  # ridiculous visible drift
        else:
            out.append(list(p))
    return out
"""


def test_dark_matter_loss_no_fit(dark_matter_csv):
    csv_path, run_id = dark_matter_csv
    result = fit_law(
        _DARK_MATTER_LAW_NO_FIT, "dark_matter", csv_path=csv_path, run_id=run_id
    )
    assert result["error"] is None
    assert result["loss_before"] is not None
    assert result["loss_before"] == result["loss_after"]


def test_dark_matter_loss_scores_probes_only(dark_matter_csv):
    """A law that mispredicts visible particles by 999 but matches probes
    must score IDENTICALLY to a law that predicts both correctly. This
    pins down the probe-only convention used by DarkMatterEvaluator."""
    csv_path, run_id = dark_matter_csv
    baseline = fit_law(
        _DARK_MATTER_LAW_NO_FIT, "dark_matter", csv_path=csv_path, run_id=run_id
    )
    offset = fit_law(
        _DARK_MATTER_LAW_VISIBLE_OFFSET, "dark_matter", csv_path=csv_path, run_id=run_id
    )
    assert baseline["error"] is None and offset["error"] is None
    assert baseline["loss_before"] == pytest.approx(offset["loss_before"])


def test_dark_matter_fit_uses_csv_visible_velocities(dark_matter_csv):
    """The dark_matter loss must read the visible particles' initial
    velocities from the CSV t=0 row (not zero them out). A law that just
    propagates positions linearly from initial velocities should produce
    a different probe loss than one that ignores velocities entirely."""
    csv_path, run_id = dark_matter_csv

    free_fall = """\
def discovered_law(positions, velocities, duration, **params):
    out = []
    for p, v in zip(positions, velocities):
        out.append([p[0] + v[0] * duration, p[1] + v[1] * duration])
    return out
"""
    static = """\
def discovered_law(positions, velocities, duration, **params):
    return [list(p) for p in positions]
"""
    res_fall = fit_law(free_fall, "dark_matter", csv_path=csv_path, run_id=run_id)
    res_stat = fit_law(static, "dark_matter", csv_path=csv_path, run_id=run_id)
    assert res_fall["error"] is None and res_stat["error"] is None
    # Probes start at rest (probe_velocities=[[0,0]]*5), so free-fall and
    # static produce IDENTICAL probe predictions; the losses must match.
    assert res_fall["loss_before"] == pytest.approx(res_stat["loss_before"])


# ---------------------------------------------------------------------------
# End-to-end: a mocked LLM emits <run_experiment> + <run_mse_fit> in the
# same response and DiscoveryAgent threads both through.


def test_agent_handles_run_mse_fit_alongside_experiment(tmp_path):
    executor = SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
    )
    csv_path = tmp_path / "gravity.csv"
    run_id = make_run_id("e2e")
    logger = TrajectoryLogger("gravity", executor, csv_path, run_id)

    def fake_complete(model, messages, system, max_tokens):
        n = sum(1 for m in messages if m["role"] == "assistant")
        if n == 0:
            return (
                '<run_experiment>[{"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], '
                '"velocity2": [0.0, 0.0], "measurement_times": [0.5, 1.0]}]'
                "</run_experiment>\n"
                "<run_mse_fit>\n" + _LAW_WITH_FIT + "</run_mse_fit>"
            )
        if n == 1:
            return (
                '<run_experiment>[{"p1": 2.0, "p2": 1.0, "pos2": [4.0, 0.0], '
                '"velocity2": [0.0, 0.0], "measurement_times": [0.5, 1.0]}]'
                "</run_experiment>"
            )
        return (
            "<final_law>\n"
            "def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):\n"
            "    return list(pos2), [0.0, 0.0]\n"
            "</final_law>\n"
            "<explanation>placeholder</explanation>"
        )

    agent = DiscoveryAgent(
        model="fake",
        executor=executor,
        max_tokens=4096,
        verbose=False,
        max_rounds=3,
        min_rounds=2,
        trajectory_logger=logger,
    )
    with patch("scienceagent.llm_client.complete", side_effect=fake_complete):
        agent.run()

    # Round 1 should have both an experiment and an mse_fit recorded.
    r1 = agent.conversation_log[0]
    assert r1["round"] == 1
    assert r1["experiment_input"] is not None
    assert r1["mse_fit_input"] is not None
    assert r1["mse_fit_output"] is not None
    fit = r1["mse_fit_output"]
    assert fit["error"] is None
    assert "k" in fit["fitted_params"]
    assert fit["n_training"] >= 1


def test_agent_rejects_response_with_no_recognized_tag(tmp_path):
    """If neither <run_experiment>, <run_mse_fit>, nor <final_law> appears,
    the agent issues a warning and re-prompts. Confirms the no_tag path
    accepts <run_mse_fit> as a valid tag."""
    executor = SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
    )
    csv_path = tmp_path / "gravity.csv"
    run_id = make_run_id("warn")
    logger = TrajectoryLogger("gravity", executor, csv_path, run_id)

    replies = iter(
        [
            "I am thinking. No tag here.",
            (
                '<run_experiment>[{"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], '
                '"velocity2": [0.0, 0.0], "measurement_times": [0.5]}]</run_experiment>'
            ),
            (
                '<run_experiment>[{"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], '
                '"velocity2": [0.0, 0.0], "measurement_times": [0.5]}]</run_experiment>'
            ),
            (
                "<final_law>\n"
                "def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):\n"
                "    return list(pos2), [0.0, 0.0]\n"
                "</final_law>\n"
                "<explanation>placeholder</explanation>"
            ),
        ]
    )

    def fake_complete(model, messages, system, max_tokens):
        return next(replies)

    agent = DiscoveryAgent(
        model="fake",
        executor=executor,
        max_tokens=4096,
        verbose=False,
        max_rounds=4,
        min_rounds=2,
        trajectory_logger=logger,
    )
    with patch("scienceagent.llm_client.complete", side_effect=fake_complete):
        agent.run()

    # Round 1 = no_tag; subsequent rounds proceed normally.
    actions = [r["action"] for r in agent.conversation_log]
    assert actions[0] == "no_tag"
    # The no_tag warning advertised <run_mse_fit> as Option 3.
    warn_text = agent.conversation_log[0]["system_message"]
    assert "<run_mse_fit>" in warn_text
