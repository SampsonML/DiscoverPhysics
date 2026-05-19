#!/usr/bin/env python
"""
Standalone analysis & plotting for a results directory.

Walks a results directory of per-trial JSONs (as produced by
`scripts/run_benchmark.py`) and writes `summary_per_model.txt` plus a
small set of plots into a separate `analysis/` subdirectory rather than
the results directory itself.

Outputs:
    summary_per_model.{txt,png,pdf}
    pareto_and_expected_passed_at_k.{png,pdf}
    per_world_passed.{png,pdf}
    world_difficulty_score.{png,pdf}
    model_world_score_heatmap.{png,pdf}

Usage:
    python scripts/run_stats.py results/yml_bench/production_r2
    python scripts/run_stats.py results/yml_bench/production_r2 --out my_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

# Per-world GT trajectory variance is shared with run_benchmark.py so the
# pass thresholds and normalized-MSE values agree across both tools.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import _WORLD_VARS  # noqa: E402

# Filenames are produced by run_benchmark.py as `<world>[_noise<σ>]_seed<N>.json`.
_SEED_RE = re.compile(r"_seed(\d+)\.json$")

_BOOTSTRAP_RESAMPLES = 5000
_BOOTSTRAP_SEED = 0  # fixed so re-aggregation is reproducible.

# Errors aggregate as geometric means; values below this are dropped (log undefined).
_GEOM_MIN = 1e-14

# A trial passes iff its (per-world variance-normalized) MSE is below
# _PASS_ERR_THRESHOLD AND its explanation score is above _PASS_SCORE_THRESHOLD.
_PASS_ERR_THRESHOLD = 0.10    # mean_pos_error / Var(GT_world) must be < this
_PASS_SCORE_THRESHOLD = 0.75  # explanation score must be >= this

# Monte Carlo draws per (model, world, k) when estimating expected_passed@k.
_EXPECTED_PASSED_SAMPLES = 1000

# Denominator used when reporting pass@k as a percentage. Hard-pinned to the
# full production world set (11 public + 11 private = 22) so percentages stay
# comparable across analysis dirs even when a run is missing some worlds.
_PASS_AT_K_TOTAL_WORLDS = 22


def _seed_pool_size(results_dir: Path, by_seed) -> int:
    """Determine how many seeds the benchmark was run with.

    Prefers `results_dir/config.yml` (`len(cfg['seeds'])`), since that records
    the intended pool size — missing seeds in the data should still count as
    fails for their slot. Falls back to scanning observed seed indices in
    `by_seed` (using `max(observed) + 1`) when config.yml is unavailable.
    Returns 0 if no information is available.
    """
    cfg_path = results_dir / "config.yml"
    if cfg_path.is_file():
        try:
            import yaml  # type: ignore
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            seeds = cfg.get("seeds")
            if isinstance(seeds, list) and seeds:
                return len(seeds)
        except (ImportError, OSError, ValueError):
            pass
    observed = {s for seed_map in by_seed.values() for s in seed_map.keys()}
    return max(observed) + 1 if observed else 0


def aggregate(results_dir: Path, analysis_dir: Path) -> Path:
    """Walk results_dir for per-trial JSONs; write summary_per_model.txt + plots."""
    by_trial: dict[tuple[str, str], list[tuple[float | None, float | None]]] = {}
    # Seed-indexed view used for worlds_passed@k. Last-write-wins per seed if
    # duplicate filenames ever appear.
    by_seed: dict[
        tuple[str, str], dict[int, tuple[float | None, float | None]]
    ] = {}

    for json_path in sorted(results_dir.rglob("*.json")):
        if json_path.name == "config.json":
            continue
        try:
            with open(json_path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        model = d.get("model")
        world = d.get("world")
        ev = d.get("evaluation") or {}
        if not model or not world or not ev:
            continue
        mpe = ev.get("mean_pos_error")
        expl = ev.get("explanation") or {}
        score = expl.get("score") if isinstance(expl, dict) else None
        by_trial.setdefault((model, world), []).append((mpe, score))
        m_seed = _SEED_RE.search(json_path.name)
        if m_seed is not None:
            by_seed.setdefault((model, world), {})[int(m_seed.group(1))] = (
                mpe, score
            )

    title = results_dir.name
    pool_size = _seed_pool_size(results_dir, by_seed)
    k_values = tuple(range(1, pool_size + 1)) if pool_size > 0 else ()
    summary_path = _write_summary(by_trial, analysis_dir, title)
    _write_per_model_summary(
        by_trial, by_seed, analysis_dir, title, pool_size, k_values
    )
    _make_plots(by_trial, by_seed, analysis_dir, title, pool_size, k_values)
    return summary_path


def _write_summary(by_trial, analysis_dir: Path, title: str) -> Path:
    """Per-(model, world) flat table; normalized MSE matches summary_per_model.txt."""
    lines = []
    lines.append(f"Benchmark summary  ({title})")
    lines.append("=" * 124)
    header = (
        f"{'model':<50} {'world':<15} {'n':>3} "
        f"{'expl_score ± SE':>22} {'norm_MSE +/− SE':>30}"
    )
    lines.append(header)
    lines.append("-" * 124)

    for (model, world), values in sorted(
        by_trial.items(), key=lambda kv: (_model_sort_key(kv[0][0]), kv[0][1])
    ):
        scores = [
            float(s)
            for _, s in values
            if isinstance(s, (int, float)) and math.isfinite(s)
        ]
        v = _WORLD_VARS.get(world)
        errs = []
        for e, _ in values:
            if not isinstance(e, (int, float)) or not math.isfinite(e):
                continue
            ev = float(e)
            if v is not None and v > 0:
                ev = ev / v
            errs.append(ev)
        n = len(values)
        score_str = _fmt_mean_bootstrap(scores)
        err_str = _fmt_geom_mean_bootstrap(errs)
        lines.append(f"{model:<50} {world:<15} {n:>3} {score_str:>22} {err_str:>30}")

    lines.append("-" * 124)
    lines.append("expl_score: explanation judge score in [0, 1], higher is better. Arithmetic mean.")
    lines.append(
        "norm_MSE: geometric mean of mean_pos_error / Var(GT_world) across seeds "
        f"(values < {_GEOM_MIN:.0e} dropped); lower is better. "
        "Per-world variances are hardcoded in run_benchmark.py."
    )
    lines.append(
        f"Format: arithmetic columns are `mean ± SE`; geometric-mean columns are "
        f"`mean +up/−down`, where up = m·(exp(SE_log)−1) and down = m·(1−exp(−SE_log)) "
        f"with SE_log estimated from {_BOOTSTRAP_RESAMPLES} log-space bootstrap "
        f"resamples (seed={_BOOTSTRAP_SEED}). The asymmetry keeps the lower bound > 0. "
        f"n=1 → no SE. 'n/a' = no successful runs."
    )

    out_path = analysis_dir / "summary.txt"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _per_model_pooled(by_trial):
    """Pool every trial equally; return {model: (errs, scores)} lists.

    Errs are divided by the per-world GT variance so the pooled geometric mean
    is comparable across worlds with different natural error scales.
    """
    pooled: dict[str, tuple[list[float], list[float]]] = {}
    for (model, world), values in by_trial.items():
        errs, scores = pooled.setdefault(model, ([], []))
        v = _WORLD_VARS.get(world)
        for e, s in values:
            if isinstance(e, (int, float)) and math.isfinite(e):
                ev = float(e)
                if v is not None and v > 0:
                    ev = ev / v
                errs.append(ev)
            if isinstance(s, (int, float)) and math.isfinite(s):
                scores.append(float(s))
    return pooled


def _write_per_model_summary(
    by_trial, by_seed, analysis_dir: Path, title: str,
    pool_size: int, k_values: tuple[int, ...],
) -> Path:
    pooled = _per_model_pooled(by_trial)
    models = sorted({m for m, _ in by_trial.keys()}, key=_model_sort_key)
    worlds = sorted({w for _, w in by_trial.keys()})
    n_worlds = len(worlds)
    passed = _passed_per_model(by_trial, models, worlds)
    n_trials = _trials_per_model(by_trial, models, worlds)
    worlds_at_k = {
        k: _worlds_passed_at_k(by_seed, models, worlds, k) for k in k_values
    }
    expected_at_k = {
        k: _expected_worlds_passed_at_k(
            by_seed, models, worlds, k, pool_size=pool_size
        )
        for k in k_values
    }
    k_col_w = max(6, len(f"{n_worlds}/{n_worlds}") + 2)
    e_col_w = max(14, len("100.00±100.00%") + 1)
    k_headers = " ".join(f"{f'@k={k}':>{k_col_w}}" for k in k_values)
    e_headers = " ".join(f"{f'E@k={k}':>{e_col_w}}" for k in k_values)
    width = 128 + (k_col_w + 1) * len(k_values) + (e_col_w + 1) * len(k_values)
    lines = []
    lines.append(f"Per-model summary  ({title})")
    lines.append("=" * width)
    header = (
        f"{'model':<50} {'n_trials':>9} {'passed':>10} "
        f"{'expl_score ± SE':>22} {'norm_MSE +/− SE':>30} "
        f"{k_headers} {e_headers}"
    )
    lines.append(header)
    lines.append("-" * width)
    for model in sorted(pooled, key=_model_sort_key):
        errs, scores = pooled[model]
        n = max(len(errs), len(scores))
        passed_str = f"{passed.get(model, 0)}/{n_trials.get(model, n)}"
        k_cells = " ".join(
            f"{f'{worlds_at_k[k].get(model, 0)}/{n_worlds}':>{k_col_w}}"
            for k in k_values
        )
        e_cells = " ".join(
            f"{f'{expected_at_k[k].get(model, (0.0, 0.0))[0] * 100.0 / _PASS_AT_K_TOTAL_WORLDS:.2f}±{expected_at_k[k].get(model, (0.0, 0.0))[1] * 100.0 / _PASS_AT_K_TOTAL_WORLDS:.2f}%':>{e_col_w}}"
            for k in k_values
        )
        lines.append(
            f"{model:<50} {n:>9} {passed_str:>10} "
            f"{_fmt_mean_bootstrap(scores):>22} {_fmt_geom_mean_bootstrap(errs):>30} "
            f"{k_cells} {e_cells}"
        )
    lines.append("-" * width)
    lines.append("Pooled across all worlds and seeds (every trial counts equally).")
    lines.append(
        f"passed: number of trials (summed over worlds & seeds) with "
        f"mean_pos_error / Var(GT_world) < {_PASS_ERR_THRESHOLD} "
        f"AND explanation_score >= {_PASS_SCORE_THRESHOLD}. "
        "Per-world variances are hardcoded in run_benchmark.py."
    )
    lines.append(
        f"@k=K: number of worlds (out of {n_worlds}) where at least one of seeds "
        "0..K-1 produced a trial-pass for that model. Missing seeds count as a fail "
        "for their slot. Monotonically non-decreasing in K."
    )
    lines.append(
        f"E@k=K: expected percentage of worlds passed when K seed positions are sampled "
        f"uniformly without replacement from a {pool_size}-seed pool, averaged "
        f"over {_EXPECTED_PASSED_SAMPLES} Monte Carlo draws (RNG seed={_BOOTSTRAP_SEED}). "
        f"Format: mean%±SEM%, scaled by 100/{_PASS_AT_K_TOTAL_WORLDS} from the raw count; SEM is the "
        "standard error of the mean across draws."
    )
    lines.append(
        "norm_MSE: geometric mean of mean_pos_error / Var(GT_world) "
        f"(values < {_GEOM_MIN:.0e} dropped); lower is better. "
        "Per-world variances are hardcoded in run_benchmark.py."
    )
    lines.append(
        f"Format: arithmetic columns are `mean ± SE`; geometric-mean columns are "
        f"`mean +up/−down`, where up = m·(exp(SE_log)−1) and down = m·(1−exp(−SE_log)) "
        f"with SE_log estimated from {_BOOTSTRAP_RESAMPLES} log-space bootstrap "
        f"resamples (seed={_BOOTSTRAP_SEED}). The asymmetry keeps the lower bound > 0. "
        f"n=1 → no SE. 'n/a' = no successful runs."
    )
    out_path = analysis_dir / "summary_per_model.txt"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


_FAMILY_CMAPS = {
    "claude": "Blues",
    "gpt": "Reds",
    "qwen": "Purples",
    "deepseek": "Greens",
}


def _model_family(model: str) -> str:
    norm = model.lower()
    for fam in ("claude", "gpt", "qwen", "deepseek"):
        if fam in norm:
            return fam
    return "other"


def _model_color_map(models, plt):
    """Stable {model: rgba} color mapping shared across every plot.

    Each model family (claude/gpt/qwen/deepseek) gets its own sequential
    colormap; models within a family are sampled at distinct lightness levels
    in canonical sort order. Anything outside the known families falls back
    to `tab10`.
    """
    by_family: dict[str, list[str]] = {}
    for m in sorted(models, key=_model_sort_key):
        by_family.setdefault(_model_family(m), []).append(m)

    colors: dict[str, tuple] = {}
    fallback = plt.get_cmap("tab10")
    fallback_idx = 0
    for fam, fam_models in by_family.items():
        if fam == "other":
            for m in fam_models:
                colors[m] = fallback(fallback_idx % 10)
                fallback_idx += 1
            continue
        cmap = plt.get_cmap(_FAMILY_CMAPS[fam])
        n = len(fam_models)
        if n == 1:
            stops = [0.75]
        else:
            stops = [0.95 - (0.95 - 0.30) * i / (n - 1) for i in range(n)]
        for m, s in zip(fam_models, stops):
            colors[m] = cmap(s)
    return colors


def _world_label(world: str) -> str:
    """Display name for a world: drop the `_easy` suffix on coulomb, underscores → spaces."""
    if world == "coulomb_easy":
        return "coulomb"
    return world.replace("_", " ")


def _short(model: str) -> str:
    """Display name for legends/axes."""
    norm = model.lower()
    if "qwen3-235b" in norm:
        return "Qwen-3.2-Instruct"
    if "qwen3.5" in norm:
        return "Qwen-3.5"
    if "claude-opus-4-7" in norm:
        return "Claude Opus 4.7"
    if "claude-sonnet-4-6" in norm:
        return "Claude Sonnet 4.6"
    if "claude-haiku-4-5" in norm:
        return "Claude Haiku 4.5"
    if "gpt-5.5" in norm:
        return "GPT-5.5"
    if "gpt-5.4" in norm:
        return "GPT-5.4"
    return model.rsplit("/", 1)[-1].replace("gpt", "GPT")


_OPUS_DIFFICULTY_MODEL = "claude-opus-4-7"


# Preferred ordering for "headline" models; everything else falls in alphabetically.
_MODEL_PRIORITY = (
    "opus-4-7",
    "sonnet-4-6",
    "haiku-4-5",
    "gpt-5-5",
    "gpt-5-4",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "deepseek-v3",
    "deepseek-r1",
    "qwen3-5",       # Qwen 3.5  (Qwen3.5-397B-A17B → "qwen3-5-...")
    "qwen3-235b",    # Qwen 3.2 Instruct
)


def _model_sort_key(model: str):
    norm = model.lower().replace(".", "-")
    for i, tag in enumerate(_MODEL_PRIORITY):
        if tag in norm:
            return (i, norm)
    return (len(_MODEL_PRIORITY), norm)


def _bootstrap_ci(values):
    """Return (mean, err_lo, err_hi) — distances to the 2.5/97.5 percentiles."""
    if not values:
        return None
    if len(values) == 1:
        return (values[0], 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_RESAMPLES, arr.size))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    m = float(arr.mean())
    return (m, m - lo, hi - m)


def _bootstrap_ci_geom(values):
    """Bootstrap CI of the geometric mean. Drops values < _GEOM_MIN."""
    arr = np.asarray([v for v in values if v >= _GEOM_MIN], dtype=float)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return (float(arr[0]), 0.0, 0.0)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_RESAMPLES, arr.size))
    boot = np.exp(np.log(arr[idx]).mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    m = float(np.exp(np.log(arr).mean()))
    return (m, m - lo, hi - m)


def _trial_passes(err, score, world: str | None) -> bool:
    """A trial passes iff (mean_pos_error / Var(GT_world)) < err threshold AND
    explanation_score >= score threshold. Falls back to raw err if world
    variance is unavailable.
    """
    if not isinstance(err, (int, float)) or not math.isfinite(err):
        return False
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        return False
    err_value = float(err)
    if world is not None:
        v = _WORLD_VARS.get(world)
        if v is not None and v > 0:
            err_value = err_value / v
    return err_value < _PASS_ERR_THRESHOLD and float(score) >= _PASS_SCORE_THRESHOLD


def _passed_per_model(by_trial, models, worlds) -> dict[str, int]:
    """Sum of trial passes across every (world, seed) for each model."""
    counts = {m: 0 for m in models}
    for (m, w), values in by_trial.items():
        if m not in counts:
            continue
        for err, score in values:
            if _trial_passes(err, score, w):
                counts[m] += 1
    return counts


def _expected_worlds_passed_at_k(
    by_seed,
    models,
    worlds,
    k: int,
    pool_size: int,
    n_samples: int = _EXPECTED_PASSED_SAMPLES,
) -> dict[str, tuple[float, float]]:
    """Return {model: (mean, sem)} of worlds_passed@k under random seed sampling.

    For each Monte Carlo draw, sample `k` distinct seed positions from
    {0..pool_size-1} (without replacement), then count how many worlds had at
    least one passing seed within that draw. Mean and standard error of the
    mean are computed over `n_samples` draws (RNG seed `_BOOTSTRAP_SEED`).
    Missing seeds count as fails for their slot.
    """
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = np.stack(
        [rng.choice(pool_size, size=k, replace=False) for _ in range(n_samples)]
    )
    out: dict[str, tuple[float, float]] = {}
    for m in models:
        per_sample_counts = np.zeros(n_samples, dtype=float)
        for w in worlds:
            seed_results = by_seed.get((m, w), {})
            passes = np.array([
                _trial_passes(*seed_results.get(s, (None, None)), w)
                for s in range(pool_size)
            ])
            if not passes.any():
                continue
            per_sample_counts += passes[samples].any(axis=1).astype(float)
        mean = float(per_sample_counts.mean())
        sem = (
            float(per_sample_counts.std(ddof=1) / math.sqrt(n_samples))
            if n_samples > 1
            else 0.0
        )
        out[m] = (mean, sem)
    return out


def _worlds_passed_at_k(by_seed, models, worlds, k: int) -> dict[str, int]:
    """Count of worlds (out of len(worlds)) passed at threshold k.

    A world is passed at k iff at least one of seed indices 0..k-1 produced a
    trial-pass for that (model, world). Missing seeds count as fails.
    """
    counts = {m: 0 for m in models}
    for m in models:
        for w in worlds:
            seed_results = by_seed.get((m, w), {})
            for seed in range(k):
                err, score = seed_results.get(seed, (None, None))
                if _trial_passes(err, score, w):
                    counts[m] += 1
                    break
    return counts


def _trials_per_model(by_trial, models, worlds) -> dict[str, int]:
    """Number of (world, seed) trials run for each model — denominator for passes."""
    return {
        m: sum(len(by_trial.get((m, w), [])) for w in worlds)
        for m in models
    }


def _make_plots(
    by_trial, by_seed, analysis_dir: Path, title: str,
    pool_size: int, k_values: tuple[int, ...],
) -> None:
    """Write the five kept plots for the benchmark."""
    if not by_trial:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots", file=sys.stderr)
        return

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })

    models = sorted({m for m, _ in by_trial.keys()}, key=_model_sort_key)
    worlds = sorted({w for _, w in by_trial.keys()})
    _make_per_model_plot(by_trial, models, analysis_dir, title, plt)
    _make_pareto_expected_combo_plot(
        by_trial, by_seed, models, worlds, analysis_dir, title, plt,
        pool_size, k_values,
    )
    _make_per_world_passed_plot(by_trial, models, worlds, analysis_dir, title, plt)
    _make_world_difficulty_plot(by_trial, worlds, analysis_dir, title, plt)
    _make_model_world_heatmap(
        by_seed, by_trial, models, worlds, analysis_dir, title, plt, k=3
    )


def _make_per_model_plot(by_trial, models, analysis_dir, title, plt) -> None:
    """One bar per model in two side-by-side panels: expl. score and Normalized MSE."""
    pooled = _per_model_pooled(by_trial)
    n_models = len(models)
    if n_models == 0:
        return
    fig_w = max(8.0, 1.2 * n_models * 2)
    fig, (ax_score, ax_err) = plt.subplots(1, 2, figsize=(fig_w, 5))
    x = np.arange(n_models)
    model_colors = _model_color_map(models, plt)
    colors = [model_colors[m] for m in models]

    score_means, score_lo, score_hi = [], [], []
    err_means, err_lo, err_hi = [], [], []
    for model in models:
        errs, scores = pooled.get(model, ([], []))
        s = _bootstrap_ci(scores)
        e = _bootstrap_ci_geom(errs)
        score_means.append(s[0] if s else np.nan)
        score_lo.append(s[1] if s else 0.0)
        score_hi.append(s[2] if s else 0.0)
        err_means.append(e[0] if e else np.nan)
        err_lo.append(e[1] if e else 0.0)
        err_hi.append(e[2] if e else 0.0)

    ax_score.bar(x, score_means, yerr=[score_lo, score_hi], color=colors, capsize=3)
    ax_err.bar(x, err_means, yerr=[err_lo, err_hi], color=colors, capsize=3)

    labels = [_short(m) for m in models]
    for ax, ax_title, ylabel in [
        (
            ax_score,
            "Explanation score (pooled across worlds)",
            "score [0, 1] ↑",
        ),
        (
            ax_err,
            "Normalized MSE (geom. mean, pooled across worlds)",
            "Normalized MSE (geom. mean) ↓",
        ),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=15)
        ax.set_ylabel(ylabel)
        ax.set_title(ax_title)
    ax_score.set_ylim(0, 1.05)
    ax_err.set_yscale("log")

    fig.suptitle(f"Per-model rollup: {title}", y=1.02)
    fig.tight_layout()
    fig.savefig(analysis_dir / "summary_per_model.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "summary_per_model.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_pareto_expected_combo_plot(
    by_trial, by_seed, models, worlds, analysis_dir, title, plt,
    pool_size: int, k_values: tuple[int, ...],
) -> None:
    """Two-panel plot: pareto (left) + expected_passed@k scatter (right).

    Single legend over the top maps model → color (shared across both panels).
    """
    if not by_trial or not models or not k_values:
        return
    from matplotlib.lines import Line2D

    pooled = _per_model_pooled(by_trial)
    model_colors = _model_color_map(models, plt)
    n_worlds = len(worlds)

    fig, (ax_pareto, ax_ek) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: pareto (geom-mean Normalized MSE, mean explanation score) per model.
    plotted_models = []
    for model in models:
        errs, scores = pooled.get(model, ([], []))
        e = _bootstrap_ci_geom(errs)
        s = _bootstrap_ci(scores)
        if e is None or s is None:
            continue
        x_mean, x_lo, x_hi = e
        y_mean, y_lo, y_hi = s
        ax_pareto.errorbar(
            x_mean, y_mean,
            xerr=[[x_lo], [x_hi]],
            yerr=[[y_lo], [y_hi]],
            fmt="o",
            color=model_colors[model],
            ecolor=model_colors[model],
            elinewidth=1.2,
            capsize=3,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
        plotted_models.append(model)

    if not plotted_models:
        plt.close(fig)
        return

    ax_pareto.set_xscale("log")
    ax_pareto.set_xlabel("Normalized MSE ↓", fontsize=18)
    ax_pareto.set_ylabel("Evaluation score (0-1) ↑", fontsize=18)
    ax_pareto.tick_params(axis="both", labelsize=13)
    ax_pareto.set_ylim(-0.05, 1.05)

    # Right: expected worlds passed @k vs k, ±SEM error bars.
    expected_at_k = {
        k: _expected_worlds_passed_at_k(
            by_seed, models, worlds, k, pool_size=pool_size
        )
        for k in k_values
    }
    xs = list(k_values)
    for model in models:
        means = [expected_at_k[k].get(model, (0.0, 0.0))[0] for k in xs]
        sems = [expected_at_k[k].get(model, (0.0, 0.0))[1] for k in xs]
        ax_ek.errorbar(
            xs, means, yerr=sems,
            marker="o", color=model_colors[model],
            ecolor=model_colors[model],
            linewidth=1.8, markersize=10,
            markeredgecolor="white", markeredgewidth=0.7,
            capsize=3, elinewidth=1.0,
        )

    ax_ek.set_xticks(xs)
    ax_ek.set_xlabel("k", fontsize=18)
    ax_ek.set_ylabel(f"Expected passed @k (out of {n_worlds}) ↑", fontsize=18)
    ax_ek.set_ylim(-0.5, n_worlds + 0.5)
    ax_ek.tick_params(axis="both", labelsize=13)
    ax_ek.grid(axis="y", linestyle=":", alpha=0.4)

    handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in plotted_models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        borderaxespad=0,
        fontsize=16,
    )

    fig.tight_layout()
    fig.savefig(
        analysis_dir / "pareto_and_expected_passed_at_k.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        analysis_dir / "pareto_and_expected_passed_at_k.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _world_score_pools(by_trial, worlds):
    """Pool normalized errors and scores across models per world.

    Errs are divided by per-world GT variance so values are normalized MSE.
    """
    errs_per_world: dict[str, list[float]] = {w: [] for w in worlds}
    scores_per_world: dict[str, list[float]] = {w: [] for w in worlds}
    for (_model, world), values in by_trial.items():
        if world not in errs_per_world:
            continue
        v = _WORLD_VARS.get(world)
        for err, score in values:
            if isinstance(err, (int, float)) and math.isfinite(err) and err >= _GEOM_MIN:
                ev = float(err)
                if v is not None and v > 0:
                    ev = ev / v
                errs_per_world[world].append(ev)
            if isinstance(score, (int, float)) and math.isfinite(score):
                scores_per_world[world].append(float(score))
    return errs_per_world, scores_per_world


def _world_order_by_score(by_trial, worlds) -> list[str]:
    """Worlds ordered by descending median explanation score (easy → hard)."""
    _errs, scores_per_world = _world_score_pools(by_trial, worlds)

    def _median_score(w: str) -> float:
        vals = scores_per_world[w]
        return float(np.median(vals)) if vals else float("-inf")

    return sorted(worlds, key=_median_score, reverse=True)


def _make_per_world_passed_plot(
    by_trial, models, worlds, analysis_dir, title, plt
) -> None:
    """Grouped bars: per-world pass rate (%) with one bar per model."""
    if not by_trial or not models or not worlds:
        return
    from matplotlib.lines import Line2D

    ordered = _world_order_by_score(by_trial, worlds)
    n_models = len(models)
    n_worlds = len(ordered)
    model_colors = _model_color_map(models, plt)

    fig_w = max(10.0, 0.9 * n_worlds * max(1, n_models) * 0.35 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    width = 0.8 / max(1, n_models)
    x = np.arange(n_worlds)

    for i, model in enumerate(models):
        pcts = []
        for w in ordered:
            values = by_trial.get((model, w), [])
            n = len(values)
            if n == 0:
                pcts.append(np.nan)
                continue
            passed = sum(1 for err, score in values if _trial_passes(err, score, w))
            pcts.append(100.0 * passed / n)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            pcts,
            width=width,
            color=model_colors[model],
            label=_short(model),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_world_label(w) for w in ordered], rotation=20, ha="right", fontsize=14
    )
    ax.set_ylabel("worlds passed (%) ↑", fontsize=16)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 105)

    handles = [
        Line2D(
            [0], [0],
            marker="s", color="w",
            markerfacecolor=model_colors[m], markeredgecolor="white",
            markersize=10, linestyle="", label=_short(m),
        )
        for m in models
    ]
    ax.legend(
        handles=handles, title="model",
        loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
    )

    fig.tight_layout()
    fig.savefig(analysis_dir / "per_world_passed.png", dpi=150, bbox_inches="tight")
    fig.savefig(analysis_dir / "per_world_passed.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_world_difficulty_plot(by_trial, worlds, analysis_dir, title, plt) -> None:
    """Per-world explanation-score violins (pooled across models), sorted easy→hard."""
    if not by_trial or not worlds:
        return

    _errs, scores_per_world = _world_score_pools(by_trial, worlds)
    ordered = _world_order_by_score(by_trial, worlds)
    score_data = [scores_per_world[w] for w in ordered if scores_per_world[w]]
    score_labels = [w for w in ordered if scores_per_world[w]]
    if not score_data:
        return

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(score_labels)), 3.5))
    parts = ax.violinplot(score_data, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("black")
        body.set_edgecolor("black")
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black")
    ax.set_xticks(range(1, len(score_labels) + 1))
    ax.set_xticklabels(
        [_world_label(w).capitalize() for w in score_labels],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=12,
    )
    ax.set_ylabel("Explanation score ↑", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(
        analysis_dir / "world_difficulty_score.png", dpi=150, bbox_inches="tight"
    )
    fig.savefig(analysis_dir / "world_difficulty_score.pdf", bbox_inches="tight")
    plt.close(fig)


def _mean_score_per_world(
    by_trial, target_models, ordered
) -> dict[tuple[str, str], tuple[float, float]]:
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for m in target_models:
        for w in ordered:
            scores = [
                s for (_e, s) in by_trial.get((m, w), [])
                if isinstance(s, (int, float)) and math.isfinite(s)
            ]
            out[(m, w)] = (float(np.mean(scores)), 0.0) if scores else (0.0, 0.0)
    return out


def _make_model_world_heatmap(
    by_seed, by_trial, models, worlds, analysis_dir, title, plt, k: int = 3
) -> None:
    """Heatmap of mean explanation score per (model, world).

    Rows = models (sort_key order). Columns = worlds (easy → hard by Opus
    median score). Cell value = mean of finite explanation scores across
    all trials for that (model, world). The k argument is unused for the
    score variant but kept for parity with other plot helpers.
    """
    if not by_trial or not models or not worlds:
        return

    _errs_per_world, scores_per_world = _world_score_pools(by_trial, worlds)
    opus_medians: dict[str, float] = {}
    for (m, w), values in by_trial.items():
        if m != _OPUS_DIFFICULTY_MODEL or w not in scores_per_world:
            continue
        vals = [
            s for (_e, s) in values
            if isinstance(s, (int, float)) and math.isfinite(s)
        ]
        if vals:
            opus_medians[w] = float(np.median(vals))
    ordered_worlds = [
        w for w in sorted(
            worlds,
            key=lambda w: opus_medians.get(w, float("-inf")),
            reverse=True,
        )
        if scores_per_world[w] and w in opus_medians
    ]
    if not ordered_worlds:
        return
    ordered_models = sorted(models, key=_model_sort_key)
    score_means = _mean_score_per_world(by_trial, ordered_models, ordered_worlds)

    n_m = len(ordered_models)
    n_w = len(ordered_worlds)
    matrix = np.array([
        [score_means.get((m, w), (np.nan, 0.0))[0] for w in ordered_worlds]
        for m in ordered_models
    ])

    fig_w = max(8.0, 0.7 * n_w + 4)
    fig_h = max(5.0, 0.45 * n_m + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    import cmasher  # noqa: F401  registers `cmr.*` colormaps with matplotlib
    cmap = plt.get_cmap("cmr.ocean")
    im = ax.imshow(
        matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto", alpha=0.75,
    )

    def _cap(s: str) -> str:
        return s[:1].upper() + s[1:] if s else s

    ax.set_xticks(np.arange(n_w))
    ax.set_xticklabels(
        [_cap(_world_label(w)) for w in ordered_worlds],
        rotation=30, ha="center", fontsize=14,
    )
    ax.set_yticks(np.arange(n_m))
    ax.set_yticklabels(
        [_cap(_short(m)) for m in ordered_models],
        fontsize=14, ha="right", va="center",
    )

    for i in range(n_m):
        for j in range(n_w):
            v = matrix[i, j]
            if not math.isfinite(v):
                continue
            text_color = "white" if v < 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean explanation score", fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    ax.set_xlabel("Physics World", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        analysis_dir / "model_world_score_heatmap.png",
        dpi=150, bbox_inches="tight",
    )
    fig.savefig(
        analysis_dir / "model_world_score_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _fmt_mean_bootstrap(values: list[float]) -> str:
    """Mean and bootstrap standard error (`_BOOTSTRAP_RESAMPLES` resamples)."""
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_RESAMPLES, arr.size))
    boot_means = arr[idx].mean(axis=1)
    se = float(boot_means.std(ddof=1))
    return f"{float(arr.mean()):.3f} ± {se:.3f}"


def _fmt_geom_mean_bootstrap(values: list[float]) -> str:
    """Geom. mean with asymmetric ±SE in raw units (derived from log-space SE).

    `up = m·(exp(SE_log) − 1)`, `down = m·(1 − exp(−SE_log))`. Both > 0, so the
    displayed lower bound `m − down` is always > 0. Drops values < _GEOM_MIN.
    """
    filtered = [float(v) for v in values if v >= _GEOM_MIN]
    if not filtered:
        return "n/a"
    if len(filtered) == 1:
        return f"{filtered[0]:.3f}"
    arr = np.asarray(filtered, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_RESAMPLES, arr.size))
    boot_log = np.log(arr[idx]).mean(axis=1)
    se_log = float(boot_log.std(ddof=1))
    m = float(np.exp(np.log(arr).mean()))
    up = m * (math.exp(se_log) - 1.0)
    down = m * (1.0 - math.exp(-se_log))
    return f"{m:.3f} +{up:.3f}/−{down:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        help="Path to a results directory (e.g. results/yml_bench/production_r2)",
    )
    parser.add_argument(
        "--out",
        default="analysis",
        help="Subdirectory name (under results_dir) for analysis output (default: analysis)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"error: {results_dir} is not a directory", file=sys.stderr)
        return 1

    analysis_dir = results_dir / args.out
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_path = aggregate(results_dir, analysis_dir)
    print(f"summary written to {summary_path}")
    print()
    print(summary_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
