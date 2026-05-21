#!/usr/bin/env python
"""
YAML-driven physics-discovery benchmark.

Generates `results/run_benchmark.py/<name>/run.sh`, executes it, then writes
`summary.txt` with mean ± std of explanation_score and mean_pos_error
per (model, world).

Usage:
    python scripts/run_benchmark.py configs/my_config.yml
    python scripts/run_benchmark.py configs/my_config.yml --no-run
    python scripts/run_benchmark.py --aggregate-only results/yml_bench/my_run
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_BOOTSTRAP_RESAMPLES = 5000
_BOOTSTRAP_SEED = 0  # fixed so summary.txt is reproducible across re-aggregation

# Errors aggregate as geometric means; values below this are dropped (log undefined).
_GEOM_MIN = 1e-14
# A (model, world) pair "passes" when these are both met across its seeds.
# Errors are normalized by per-world GT trajectory variance before thresholding,

# --- the pass thresholds for a world
_PASS_ERR_THRESHOLD = 0.1  # mean_pos_error / Var(GT_world) must be < this
_PASS_SCORE_THRESHOLD = 0.75  # arithmetic mean of explanation score must be >= this

# --- for fair MSE comparison divide MSE by variance of each world trajectories
_WORLD_VARS: dict[str, float] = {
    "circle": 6.596,
    "confinement": 3.636,
    "coulomb_easy": 4.244,
    "dark_matter": 63.303,
    "diffusion": 5.878,
    "ether": 21.259,
    "ether_hard": 33.520,
    "extra_dimensions": 4.248,
    "force_geography": 4.074,
    "fractional": 5.721,
    "gravity": 4.283,
    "gravity_hard": 16.172,
    "hubble": 41.189,
    "hubble_hard": 227.908,
    "mond": 28.380,
    "non_symmetric": 46.759,
    "oscillator": 6.332,
    "running_coupling": 4.482,
    "species": 1.985,
    "three_species": 28.717,
    "yukawa": 5.677,
}

try:
    import yaml
except ImportError:
    sys.stderr.write("error: PyYAML is not installed. Run `pip install pyyaml`.\n")
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")

    for k in ("models", "critic", "max_rounds", "worlds", "seeds"):
        if k not in cfg:
            raise ValueError(f"{path}: missing required field `{k}`")

    critic = cfg["critic"]
    if isinstance(critic, bool):
        critic = "on" if critic else "off"
    elif isinstance(critic, str):
        critic = critic.lower()
    if critic not in ("on", "off"):
        raise ValueError("critic must be 'on'/'off' (or true/false)")
    cfg["critic"] = critic

    random_experiments = cfg.get("random_experiments", False)
    if isinstance(random_experiments, bool):
        random_experiments = "on" if random_experiments else "off"
    elif isinstance(random_experiments, str):
        random_experiments = random_experiments.lower()
    if random_experiments not in ("on", "off"):
        raise ValueError("random_experiments must be 'on'/'off' (or true/false)")
    cfg["random_experiments"] = random_experiments

    no_mse = cfg.get("no_mse", False)
    if isinstance(no_mse, bool):
        no_mse = "on" if no_mse else "off"
    elif isinstance(no_mse, str):
        no_mse = no_mse.lower()
    if no_mse not in ("on", "off"):
        raise ValueError("no_mse must be 'on'/'off' (or true/false)")
    if no_mse == "on" and random_experiments == "on":
        raise ValueError("no_mse and random_experiments are mutually exclusive")
    cfg["no_mse"] = no_mse
    if not isinstance(cfg["models"], list) or not cfg["models"]:
        raise ValueError("models must be a non-empty list")
    if not isinstance(cfg["worlds"], list) or not cfg["worlds"]:
        raise ValueError("worlds must be a non-empty list")
    if not isinstance(cfg["seeds"], list) or not cfg["seeds"]:
        raise ValueError("seeds must be a non-empty list")
    if not isinstance(cfg["max_rounds"], int):
        raise ValueError("max_rounds must be an int")

    cfg.setdefault("noise_std", 0.0)
    raw_noise = cfg["noise_std"]
    if isinstance(raw_noise, (int, float)):
        noise_levels = [float(raw_noise)]
    elif (
        isinstance(raw_noise, list)
        and raw_noise
        and all(isinstance(v, (int, float)) for v in raw_noise)
    ):
        noise_levels = [float(v) for v in raw_noise]
    else:
        raise ValueError("noise_std must be a number or a non-empty list of numbers")
    cfg["noise_std"] = noise_levels

    # noise_frac: signal-relative noise. Per world, the absolute sigma handed to
    # run_discovery is `frac * sqrt(_WORLD_VARS[world])`, i.e. the noise std as a
    # fraction of the per-world ground-truth position std — the same Var(GT) the
    # pass metric normalizes MSE by. Mutually exclusive with an absolute noise_std.
    raw_frac = cfg.get("noise_frac")
    if raw_frac is None:
        cfg["noise_mode"] = "abs"
        cfg["noise_frac"] = []
    else:
        if isinstance(raw_frac, (int, float)):
            frac_levels = [float(raw_frac)]
        elif (
            isinstance(raw_frac, list)
            and raw_frac
            and all(isinstance(v, (int, float)) for v in raw_frac)
        ):
            frac_levels = [float(v) for v in raw_frac]
        else:
            raise ValueError(
                "noise_frac must be a number or a non-empty list of numbers"
            )
        if any(v < 0.0 for v in frac_levels):
            raise ValueError("noise_frac values must be >= 0")
        if noise_levels != [0.0]:
            raise ValueError(
                "noise_frac and a nonzero noise_std are mutually exclusive — "
                "set only one"
            )
        missing = sorted(w for w in cfg["worlds"] if w not in _WORLD_VARS)
        if missing:
            raise ValueError(
                "noise_frac needs a Var(GT) for every world, but _WORLD_VARS in "
                f"run_benchmark.py is missing: {', '.join(missing)}"
            )
        cfg["noise_mode"] = "frac"
        cfg["noise_frac"] = frac_levels

    cfg.setdefault("critic_model", "claude-haiku-4-5-20251001")
    cfg.setdefault("judge_model", "claude-opus-4-6")
    cfg.setdefault("name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return cfg


def generate_bash_script(cfg: dict, out_root: Path) -> Path:
    """Write run.sh inside out_root and return its path."""
    models_str = " ".join(f'"{m}"' for m in cfg["models"])
    worlds_str = " ".join(f'"{w}"' for w in cfg["worlds"])
    seeds_str = " ".join(str(s) for s in cfg["seeds"])

    frac_mode = cfg["noise_mode"] == "frac"
    # In frac mode the NOISES array carries the *fraction* levels (used only for
    # output tags / labels, world-independent); the absolute sigma is derived
    # per world at runtime as `frac * WORLD_SIGMA[world]`.
    noise_drive = cfg["noise_frac"] if frac_mode else cfg["noise_std"]
    noise_str = " ".join(f"{v:.4g}" for v in noise_drive)
    multi_noise = len(noise_drive) > 1
    if frac_mode:
        world_sigma_pairs = " ".join(
            f'["{w}"]="{_WORLD_VARS[w] ** 0.5:.6f}"' for w in cfg["worlds"]
        )
        world_sigma_decl = f"declare -A WORLD_SIGMA=( {world_sigma_pairs} )"
    else:
        world_sigma_decl = "declare -A WORLD_SIGMA=( )"

    script = f"""#!/usr/bin/env bash
# Running DiscoverBench! 
# Auto-generated by scripts/run_benchmark.py for benchmark "{cfg['name']}".
# Do not edit by hand — regenerate from configs/<...>.yml.
set -uo pipefail

REPO_ROOT="{REPO_ROOT}"
cd "${{REPO_ROOT}}"

if [[ -f "${{REPO_ROOT}}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${{REPO_ROOT}}/.env"
  set +a
else
  echo "WARNING: ${{REPO_ROOT}}/.env not found — API-key-backed calls will fail." >&2
fi

OUT_ROOT="{out_root}"

MODELS=( {models_str} )
WORLDS=( {worlds_str} )
SEEDS=( {seeds_str} )
NOISES=( {noise_str} )
CRITIC="{cfg['critic']}"
CRITIC_MODEL="{cfg['critic_model']}"
JUDGE_MODEL="{cfg['judge_model']}"
MAX_ROUNDS={cfg['max_rounds']}
RANDOM_EXPERIMENTS="{cfg['random_experiments']}"
NO_MSE="{cfg['no_mse']}"
MULTI_NOISE={1 if multi_noise else 0}
NOISE_MODE="{cfg['noise_mode']}"
{world_sigma_decl}

N_TOTAL=$(( ${{#MODELS[@]}} * ${{#WORLDS[@]}} * ${{#SEEDS[@]}} * ${{#NOISES[@]}} ))

echo "=========================================="
echo "YAML benchmark: {cfg['name']}"
echo "  output : ${{OUT_ROOT}}"
echo "  models : ${{MODELS[*]}}"
echo "  worlds : ${{WORLDS[*]}}"
echo "  seeds  : ${{SEEDS[*]}}"
echo "  critic : ${{CRITIC}} (model: ${{CRITIC_MODEL}})"
echo "  judge  : ${{JUDGE_MODEL}}"
echo "  rounds : ${{MAX_ROUNDS}}"
if [[ "${{NOISE_MODE}}" == "frac" ]]; then
  echo "  noise  : σ/σ_signal (frac of per-world sqrt(Var GT)) = ${{NOISES[*]}}"
else
  echo "  noise  : σ=${{NOISES[*]}} (absolute)"
fi
echo "  random : ${{RANDOM_EXPERIMENTS}}"
echo "  no_mse : ${{NO_MSE}}"
echo "  total trials : ${{N_TOTAL}}"
echo "=========================================="

trial=0
t_start=$(date +%s)

for model in "${{MODELS[@]}}"; do
  safe_model="${{model//\\//_}}"
  for world in "${{WORLDS[@]}}"; do
    for noise in "${{NOISES[@]}}"; do
      for seed in "${{SEEDS[@]}}"; do
        trial=$((trial + 1))
        if [[ "${{MULTI_NOISE}}" == "1" ]]; then
          noise_tag="_noise${{noise}}"
        else
          noise_tag=""
        fi
        base="${{OUT_ROOT}}/${{safe_model}}/${{world}}${{noise_tag}}_seed${{seed}}"
        mkdir -p "$(dirname "${{base}}")"

        # Resolve the absolute sigma passed to run_discovery. In frac mode the
        # loop value is a fraction of the per-world signal std sqrt(Var GT).
        if [[ "${{NOISE_MODE}}" == "frac" ]]; then
          world_sigma="${{WORLD_SIGMA[$world]:-}}"
          if [[ -z "${{world_sigma}}" ]]; then
            echo "    FAILED — no WORLD_SIGMA for world '${{world}}' (missing _WORLD_VARS entry)"
            continue
          fi
          actual_noise=$(awk -v f="${{noise}}" -v s="${{world_sigma}}" \\
            'BEGIN {{ printf "%.6f", f * s }}')
        else
          actual_noise="${{noise}}"
        fi

        critic_flags=()
        if [[ "${{CRITIC}}" == "on" ]]; then
          critic_flags=(--use-critic --critic-model "${{CRITIC_MODEL}}")
        fi

        random_flags=()
        if [[ "${{RANDOM_EXPERIMENTS}}" == "on" ]]; then
          random_flags=(--random-experiments --random-seed "${{seed}}")
        fi

        no_mse_flags=()
        if [[ "${{NO_MSE}}" == "on" ]]; then
          no_mse_flags=(--no-mse)
        fi

        t_trial_start=$(date +%s)
        printf "[%3d/%3d] model=%s world=%s noise=%s (σ=%s) seed=%s\\n" \\
          "${{trial}}" "${{N_TOTAL}}" "${{model}}" "${{world}}" "${{noise}}" "${{actual_noise}}" "${{seed}}"

        python ScienceAgent/run_discovery.py \\
          --model "${{model}}" \\
          --world "${{world}}" \\
          --noise-std "${{actual_noise}}" \\
          --noise-seed "${{seed}}" \\
          --max-rounds "${{MAX_ROUNDS}}" \\
          --judge-model "${{JUDGE_MODEL}}" \\
          --store_output "${{base}}" \\
          --quiet \\
          ${{critic_flags[@]+"${{critic_flags[@]}}"}} \\
          ${{random_flags[@]+"${{random_flags[@]}}"}} \\
          ${{no_mse_flags[@]+"${{no_mse_flags[@]}}"}} \\
          > "${{base}}.stdout.log" 2>&1
        rc=$?
        t_trial_end=$(date +%s)

        if [[ ${{rc}} -ne 0 ]]; then
          echo "    FAILED (rc=${{rc}}) — see ${{base}}.stdout.log"
        else
          printf "    done in %ds\\n" "$((t_trial_end - t_trial_start))"
        fi
      done
    done
  done
done

t_end=$(date +%s)
echo "=========================================="
echo "Finished ${{trial}} trials in $((t_end - t_start))s"
echo "=========================================="
"""
    script_path = out_root / "run.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def aggregate(out_root: Path) -> Path:
    """Walk out_root for per-trial JSONs and write summary.txt."""
    by_trial: dict[tuple[str, str], list[tuple[float | None, float | None]]] = {}

    for json_path in sorted(out_root.rglob("*.json")):
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

    lines = []
    lines.append(f"Benchmark summary  ({out_root.name})")
    lines.append("=" * 116)
    header = (
        f"{'model':<50} {'world':<15} {'n':>3} "
        f"{'expl_score [95% CI]':>22} {'norm_MSE [95% CI]':>22}"
    )
    lines.append(header)
    lines.append("-" * 116)

    for (model, world), values in sorted(by_trial.items()):
        scores = [
            float(s)
            for _, s in values
            if isinstance(s, (int, float)) and math.isfinite(s)
        ]
        errs = _normalized_errs(values, world)
        n = len(values)
        score_str = _fmt_mean_bootstrap(scores)
        err_str = _fmt_geom_mean_bootstrap(errs)
        lines.append(f"{model:<50} {world:<15} {n:>3} {score_str:>22} {err_str:>22}")

    lines.append("-" * 116)
    lines.append(
        "expl_score: explanation judge score in [0, 1], higher is better. Arithmetic mean."
    )
    lines.append(
        "norm_MSE: geometric mean of mean_pos_error / Var(GT_world) across seeds "
        f"(values < {_GEOM_MIN:.0e} dropped); lower is better. "
        "Per-world variances are hardcoded in run_benchmark.py."
    )
    lines.append(
        f"Format: point estimate [2.5%, 97.5%] from {_BOOTSTRAP_RESAMPLES} bootstrap "
        f"resamples (seed={_BOOTSTRAP_SEED}). n=1 → no CI. 'n/a' = no successful runs."
    )

    out_path = out_root / "summary.txt"
    out_path.write_text("\n".join(lines) + "\n")
    _write_per_model_summary(by_trial, out_root)
    return out_path


def _per_model_pooled(by_trial):
    """Pool every trial equally; return {model: (errs, scores)} lists.

    Errors are divided by per-world GT variance so the pooled geometric mean
    is a normalized MSE comparable across worlds.
    """
    pooled: dict[str, tuple[list[float], list[float]]] = {}
    for (model, world), values in by_trial.items():
        errs, scores = pooled.setdefault(model, ([], []))
        errs.extend(_normalized_errs(values, world))
        for _, s in values:
            if isinstance(s, (int, float)) and math.isfinite(s):
                scores.append(float(s))
    return pooled


def _write_per_model_summary(by_trial, out_root: Path) -> Path:
    pooled = _per_model_pooled(by_trial)
    models = sorted({m for m, _ in by_trial.keys()})
    worlds = sorted({w for _, w in by_trial.keys()})
    passed = _passed_per_model(by_trial, models, worlds)
    n_trials = _trials_per_model(by_trial, models, worlds)
    lines = []
    lines.append(f"Per-model summary  ({out_root.name})")
    lines.append("=" * 124)
    header = (
        f"{'model':<50} {'n_trials':>9} {'passed':>10} "
        f"{'expl_score [95% CI]':>22} {'norm_MSE [95% CI]':>26}"
    )
    lines.append(header)
    lines.append("-" * 124)
    for model in sorted(pooled):
        errs, scores = pooled[model]
        n = max(len(errs), len(scores))
        passed_str = f"{passed.get(model, 0)}/{n_trials.get(model, n)}"
        lines.append(
            f"{model:<50} {n:>9} {passed_str:>10} "
            f"{_fmt_mean_bootstrap(scores):>22} {_fmt_geom_mean_bootstrap(errs):>26}"
        )
    lines.append("-" * 124)
    lines.append("Pooled across all worlds and seeds (every trial counts equally).")
    lines.append(
        f"passed: number of trials (summed over worlds & seeds) with "
        f"mean_pos_error / Var(GT_world) < {_PASS_ERR_THRESHOLD} "
        f"AND explanation_score >= {_PASS_SCORE_THRESHOLD}. "
        "Per-world variances are hardcoded in run_benchmark.py."
    )
    lines.append(
        "norm_MSE: geometric mean of mean_pos_error / Var(GT_world) "
        f"(values < {_GEOM_MIN:.0e} dropped); lower is better."
    )
    lines.append(
        f"Format: point estimate [2.5%, 97.5%] from {_BOOTSTRAP_RESAMPLES} bootstrap "
        f"resamples (seed={_BOOTSTRAP_SEED}). n=1 → no CI. 'n/a' = no successful runs."
    )
    out_path = out_root / "summary_per_model.txt"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _normalized_errs(values, world: str) -> list[float]:
    """Extract finite errs from `values` and divide by Var(GT_world).

    Falls back to raw errs if the per-world variance is unavailable or zero.
    """
    v = _WORLD_VARS.get(world)
    if v is not None and v > 0:
        return [
            float(e) / v
            for e, _ in values
            if isinstance(e, (int, float)) and math.isfinite(e)
        ]
    return [
        float(e) for e, _ in values if isinstance(e, (int, float)) and math.isfinite(e)
    ]


def _trial_passes(err, score, world: str | None) -> bool:
    """A single trial passes iff (mean_pos_error / Var(GT_world)) < err threshold
    AND explanation_score >= score threshold. Falls back to raw err if world
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


def _trials_per_model(by_trial, models, worlds) -> dict[str, int]:
    """Number of (world, seed) trials run for each model — denominator for passes."""
    return {m: sum(len(by_trial.get((m, w), [])) for w in worlds) for m in models}


def _fmt_mean_bootstrap(values: list[float]) -> str:
    """Mean and 95% bootstrap CI of the mean (`_BOOTSTRAP_RESAMPLES` resamples)."""
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_RESAMPLES, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return f"{float(arr.mean()):.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_geom_mean_bootstrap(values: list[float]) -> str:
    """Geom. mean and 95% bootstrap CI of the geom. mean. Drops values < _GEOM_MIN."""
    filtered = [float(v) for v in values if v >= _GEOM_MIN]
    if not filtered:
        return "n/a"
    if len(filtered) == 1:
        return f"{filtered[0]:.3f}"
    arr = np.asarray(filtered, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_RESAMPLES, arr.size))
    boot = np.exp(np.log(arr[idx]).mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    m = float(np.exp(np.log(arr).mean()))
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", nargs="?", help="Path to YAML config")
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Generate run.sh but do not execute it",
    )
    parser.add_argument(
        "--aggregate-only",
        metavar="DIR",
        help="Skip generation/run; aggregate an existing output dir",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Override output root (default: results/yml_bench/<name>)",
    )
    args = parser.parse_args()

    if args.aggregate_only:
        out_root = Path(args.aggregate_only)
        if not out_root.is_dir():
            print(f"error: {out_root} is not a directory", file=sys.stderr)
            return 1
        path = aggregate(out_root)
        print(f"summary written to {path}")
        print()
        print(path.read_text())
        return 0

    if not args.config:
        parser.error("config is required (unless --aggregate-only)")

    cfg = load_config(Path(args.config))
    out_root = (
        Path(args.out_root)
        if args.out_root
        else REPO_ROOT / "results" / "yml_bench" / cfg["name"]
    )
    out_root.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, out_root / "config.yml")
    script_path = generate_bash_script(cfg, out_root)
    print(f"Wrote {script_path}")

    if args.no_run:
        print("(--no-run requested; skipping execution)")
        return 0

    print(f"Executing {script_path} ...\n")
    rc = subprocess.run(["bash", str(script_path)]).returncode
    if rc != 0:
        print(f"\nrun.sh exited with {rc}; partial results may exist", file=sys.stderr)

    summary_path = aggregate(out_root)
    print(f"\nsummary written to {summary_path}")
    print(f"--- {summary_path.name} ---")
    print(summary_path.read_text())
    return rc


if __name__ == "__main__":
    sys.exit(main())
