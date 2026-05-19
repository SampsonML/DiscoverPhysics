#!/usr/bin/env python
"""
CLI runner for a single physics-discovery episode.
"""

import argparse
import json
import os
from pathlib import Path

# Auto-load .env from project root if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_path)


def main():
    parser = argparse.ArgumentParser(description="Run a physics discovery episode.")
    parser.add_argument("--model", default="claude-opus-4-6", help="LLM model string")
    parser.add_argument("--world", default="gravity", help="World name (see worlds.py)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens (default: 16384 for reasoning models, 8192 otherwise)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-round output"
    )
    parser.add_argument(
        "--show-experiment-output",
        action="store_true",
        help="Print simulator experiment output each round",
    )
    parser.add_argument(
        "--output", default=None, help="Write results JSON to this file"
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Save trajectory comparison plot to this path (e.g. plot.png)",
    )
    parser.add_argument(
        "--store_output",
        default=None,
        help="Base path for full run logs: writes <path>.json and <path>.txt",
    )
    parser.add_argument(
        "--use-critic", action="store_true", help="Enable supervisor critic agent"
    )
    parser.add_argument(
        "--critic-model",
        default="claude-haiku-4-5-20251001",
        help="Model for the critic agent (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-opus-4-6",
        help="Model used by the LLM-judge that scores the agent's prose "
        "explanation against the world's optimal_explanation "
        "(default: claude-opus-4-6).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Std-dev of Gaussian observation noise added to particle "
        "positions returned to the agent. Velocities are also noised, with "
        "std noise_std/Δt (Δt = median sampling cadence). Evaluator ground "
        "truth stays clean. 0.0 disables noise.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible noise.",
    )
    parser.add_argument(
        "--engine",
        choices=("field", "nbody"),
        default="nbody",
        help="Simulation backend: 'nbody' (default; direct O(N²) "
        "NBodySampler with Yoshida-4 symplectic; more accurate for "
        "static-PDE worlds) or 'field' (FFT FieldSampler; required for "
        "the diffusion/wave worlds).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Override the agent's max experimentation rounds "
        "(default: 10 as set in DiscoveryAgent).",
    )
    parser.add_argument(
        "--random-experiments",
        action="store_true",
        help="Run in random-experiments mode: instead of the "
        "agent proposing experiments, one experiment per "
        "round is drawn uniformly from the documented "
        "parameter ranges and executed automatically. "
        "The agent only analyses the data. Uses the "
        "matching *_random.md system prompt for the world.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="RNG seed for the random-experiment sampler. "
        "If --random-experiments is set but this is "
        "omitted, falls back to --noise-seed (if any).",
    )
    parser.add_argument(
        "--no-trajectory-csv",
        action="store_true",
        help="Disable per-experiment trajectory CSV logging "
        "(default: log to results/trajectories/<world>.csv).",
    )
    parser.add_argument(
        "--no-mse",
        action="store_true",
        help="Disable mid-round MSE fitting: the <run_mse_fit> "
        "tool is hidden from the prompt and any <run_mse_fit> tag "
        "the agent emits is rejected with an error message. "
        "End-of-mission fitting via fit_parameters() in <final_law> "
        "still runs as usual. Uses PhysicsSchool/prompts/_template_no_mse.md.",
    )
    args = parser.parse_args()

    if args.no_mse and args.random_experiments:
        parser.error("--no-mse and --random-experiments are mutually exclusive")

    from scienceagent.worlds import get_world
    from scienceagent.agent import DiscoveryAgent
    from scienceagent.random_experiments import make_random_generator
    from scienceagent.evaluator import (
        Evaluator,
        CircleEvaluator,
        SpeciesEvaluator,
        ThreeSpeciesEvaluator,
        DarkMatterEvaluator,
        EtherEvaluator,
        HubbleEvaluator,
        CoulombHardEvaluator,
        ExplanationJudge,
        _extract_training_trajectories,
        clean_law_source,
    )

    critic = None
    # the critic helps ensure the rules (how to run experiments) are followed
    # it also provides input on if the past experiment was "useful"
    if args.use_critic:
        from scienceagent.critic import CriticAgent

        critic = CriticAgent(model=args.critic_model)

    print(f"World : {args.world}")
    print(f"Engine: {args.engine}")
    print(f"Model : {args.model}")
    if critic:
        print(f"Critic: {args.critic_model}")
    if args.noise_std > 0:
        seed_str = f", seed={args.noise_seed}" if args.noise_seed is not None else ""
        print(f"Noise : Gaussian σ={args.noise_std} on positions{seed_str}")
    if args.random_experiments:
        random_seed = (
            args.random_seed if args.random_seed is not None else args.noise_seed
        )
        seed_str = f", seed={random_seed}" if random_seed is not None else ""
        print(f"Mode  : random experiments (one per round{seed_str})")
    print()

    world = get_world(
        args.world,
        engine=args.engine,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
    )
    executor = world["executor"]
    mission = world["mission"]
    true_law = world["true_law"]
    true_law_title = world["true_law_title"]
    optimal_explanation = world.get("optimal_explanation", "")
    explanation_rubric = world.get("explanation_rubric", "")
    system_prompt_path = world["system_prompt"]
    instructions_path = world["instructions"]

    random_generator = None
    if args.random_experiments:
        random_seed = (
            args.random_seed if args.random_seed is not None else args.noise_seed
        )
        random_generator = make_random_generator(args.world, seed=random_seed)
        # All worlds share one random-mode master template; the per-topology
        # instructions block is substituted in by the agent's loader.
        system_prompt_path = "PhysicsSchool/prompts/_template_random.md"
    elif args.no_mse:
        # Same per-topology instructions as interactive mode, but the prompt
        # explicitly tells the agent there is no mid-round MSE fitting and
        # the agent loop rejects <run_mse_fit> tags.
        system_prompt_path = "PhysicsSchool/prompts/_template_no_mse.md"

    # Reasoning models need more output tokens for chain-of-thought + XML tags
    max_tokens = args.max_tokens
    if max_tokens is None:
        _reasoning_prefixes = ("azure/gpt-5.4-pro", "o1", "o3")
        if any(args.model.startswith(p) for p in _reasoning_prefixes):
            max_tokens = 16384
        else:
            max_tokens = 8192

    trajectory_logger = None
    if not args.no_trajectory_csv:
        from scienceagent.trajectory_logger import TrajectoryLogger, make_run_id

        repo_root = Path(__file__).resolve().parent.parent
        csv_path = repo_root / "results" / "trajectories" / f"{args.world}.csv"
        run_id = make_run_id(args.model)
        trajectory_logger = TrajectoryLogger(
            world=args.world,
            executor=executor,
            csv_path=csv_path,
            run_id=run_id,
        )
        # Always remove the per-run trajectory CSV on exit. The CSV is only
        # consumed by the mid-round <run_mse_fit> tool during the agent
        # loop; by the time the process is exiting, it has served its
        # purpose. atexit fires on normal exit *and* after exceptions
        # propagate up, so a crashed run still leaves no leftover file.
        import atexit

        def _cleanup_trajectory_csv(path=csv_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        atexit.register(_cleanup_trajectory_csv)
        print(
            f"Trajectory CSV: {csv_path}  (run_id={run_id})  " f"[auto-removed on exit]"
        )

    agent_kwargs = dict(
        model=args.model,
        executor=executor,
        mission=mission,
        max_tokens=max_tokens,
        verbose=not args.quiet,
        show_experiment_output=args.show_experiment_output,
        system_prompt_path=system_prompt_path,
        instructions_path=instructions_path,
        law_stub=world["law_stub"],
        experiment_format=world["experiment_format"],
        critic=critic,
        random_experiments=args.random_experiments,
        random_generator=random_generator,
        trajectory_logger=trajectory_logger,
        no_mse=args.no_mse,
    )
    if args.max_rounds is not None:
        agent_kwargs["max_rounds"] = args.max_rounds
    agent = DiscoveryAgent(**agent_kwargs)

    law_source = agent.run()

    if law_source is None:
        print("\n[No final law submitted]")
        return

    print("\n" + "=" * 60)
    print("Discovered law:")
    print("=" * 60)
    print(law_source)

    print("\n" + "=" * 60)
    print("Evaluation:")
    print("=" * 60)
    if args.world == "circle":
        evaluator = CircleEvaluator(executor)
    elif args.world == "three_species":
        evaluator = ThreeSpeciesEvaluator(executor)
    elif args.world == "dark_matter":
        evaluator = DarkMatterEvaluator(executor)
    elif args.world == "ether":
        evaluator = EtherEvaluator(executor)
    elif args.world == "hubble":
        evaluator = HubbleEvaluator(executor)
    elif world.get("evaluator_class") is not None:
        evaluator = world["evaluator_class"](executor)
    else:
        evaluator = Evaluator(executor)

    # Continuous-parameter worlds can fit free parameters declared by the
    # agent's optional `fit_parameters()` function. Pass the training
    # trajectories the agent collected during discovery as the fit set;
    # structural worlds (species, three_species, dark_matter) skip this.
    _FIT_WORLDS = {
        "gravity",
        "yukawa",
        "fractional",
        "diffusion",
        "wave",
        "oscillator",
        "extra_dimensions",
        "circle",
        "ether",
        "hubble",
    }
    eval_kwargs = {"verbose": True}
    if args.world in _FIT_WORLDS:
        eval_kwargs["training_trajectories"] = _extract_training_trajectories(
            agent.conversation_log
        )
    results = evaluator.evaluate(law_source, **eval_kwargs)

    # Independent text-based metric: judge the agent's prose explanation
    # against the world's ground-truth optimal_explanation.
    print("\n" + "=" * 60)
    print("Explanation evaluation:")
    print("=" * 60)
    print(f"Agent explanation: {agent.discovered_explanation or '(none submitted)'}")
    print()
    # Judge default: a strong model from a different family/provider than
    # any of the benchmarked agent models, so no agent grades its own
    # explanations.  Override via --judge-model.
    judge_model = args.judge_model
    judge = ExplanationJudge(judge_model=judge_model)
    explanation_result = judge.score(
        agent_explanation=agent.discovered_explanation,
        optimal_explanation=optimal_explanation,
        rubric=explanation_rubric,
        verbose=True,
    )
    results["explanation"] = {
        "agent_explanation": agent.discovered_explanation,
        "optimal_explanation": optimal_explanation,
        "judge_model": judge_model,
        **explanation_result,
    }

    if args.plot:
        # will clean this up and refactor the plotting code soon
        base = args.plot
        plot_path = base + "_plot.png"
        law_path = base + "_law.png"
        if args.world in ("circle", "mond"):
            # mond shares circle's 11-particle (1 source + 10 ring)
            # layout and trajectory schema, so the circle plotters render
            # it correctly.
            _plot_circle_trajectories(
                results["trajectories"], args.world, args.model, plot_path
            )
            _plot_circle_law(
                clean_law_source(law_source), args.world, args.model, law_path
            )
        elif args.world in ("species", "gravity_hard", "non_symmetric", "coulomb_hard"):
            _plot_species_trajectories(
                results["trajectories"], args.world, args.model, plot_path
            )
            _plot_law(clean_law_source(law_source), args.world, args.model, law_path)
        elif args.world == "three_species":
            _plot_three_species_trajectories(
                results["trajectories"], args.world, args.model, plot_path
            )
            _plot_law(clean_law_source(law_source), args.world, args.model, law_path)
        elif args.world == "dark_matter":
            _plot_dark_matter_trajectories(
                results["trajectories"], args.world, args.model, plot_path
            )
            _plot_law(clean_law_source(law_source), args.world, args.model, law_path)
        elif args.world in ("ether", "ether_hard"):
            _plot_ether_trajectories(
                results["trajectories"], args.world, args.model, plot_path
            )
            _plot_law(clean_law_source(law_source), args.world, args.model, law_path)
        elif args.world in ("hubble", "hubble_hard"):
            # Hubble shares the 26-particle anchor + ring + probes layout, so
            # the ether plotter renders correctly for it too.
            _plot_ether_trajectories(
                results["trajectories"], args.world, args.model, plot_path
            )
            _plot_law(clean_law_source(law_source), args.world, args.model, law_path)
        else:
            _plot_trajectories(
                results["trajectories"], args.world, args.model, plot_path
            )
            _plot_law(clean_law_source(law_source), args.world, args.model, law_path)

    if args.output:
        out = {
            "world": args.world,
            "engine": args.engine,
            "model": args.model,
            "noise_std": args.noise_std,
            "noise_seed": args.noise_seed,
            "law": law_source,
            "evaluation": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output}")

    if args.store_output:
        import datetime

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        _write_run_json(
            args.store_output + ".json",
            args.world,
            args.model,
            timestamp,
            agent,
            law_source,
            results,
            noise_std=args.noise_std,
            noise_seed=args.noise_seed,
        )
        _write_run_txt(
            args.store_output + ".txt",
            args.world,
            args.model,
            timestamp,
            agent,
            law_source,
            results,
            noise_std=args.noise_std,
            noise_seed=args.noise_seed,
        )
        print(f"\nFull run log written to {args.store_output}.json / .txt")


def _write_run_json(
    path,
    world,
    model,
    timestamp,
    agent,
    law_source,
    evaluation,
    noise_std=0.0,
    noise_seed=None,
):
    """Write the full structured run log as JSON."""
    data = {
        "world": world,
        "model": model,
        "timestamp": timestamp,
        "noise_std": noise_std,
        "noise_seed": noise_seed,
        "max_rounds": agent.max_rounds,
        "mission": agent.mission,
        "rounds": agent.conversation_log,
        "final_law": law_source,
        "evaluation": evaluation,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_run_txt(
    path,
    world,
    model,
    timestamp,
    agent,
    law_source,
    evaluation,
    noise_std=0.0,
    noise_seed=None,
):
    """Write the full run log as a human-readable plain-text file."""
    lines = []
    sep = "=" * 60

    noise_line = f"σ={noise_std}" + (
        f", seed={noise_seed}" if noise_seed is not None else ""
    )
    lines += [
        sep,
        "PHYSICS DISCOVERY RUN",
        sep,
        f"World     : {world}",
        f"Model     : {model}",
        f"Timestamp : {timestamp}",
        f"Noise     : {noise_line}",
        f"Mission   : {agent.mission or '(none)'}",
        "",
    ]

    for entry in agent.conversation_log:
        lines += [sep, f"ROUND {entry['round']}", sep]

        if entry.get("system_message"):
            lines += ["[System]", entry["system_message"], ""]

        if entry.get("llm_reply"):
            lines += ["[LLM Reply]", entry["llm_reply"], ""]

        action = entry.get("action")
        if action == "experiment":
            if entry.get("experiment_input") is not None:
                lines += [
                    "[Experiment Input]",
                    json.dumps(entry["experiment_input"], indent=2),
                    "",
                ]
            if entry.get("experiment_output") is not None:
                lines += [
                    "[Experiment Output]",
                    json.dumps(entry["experiment_output"], indent=2),
                    "",
                ]
            if entry.get("experiment_error"):
                lines += ["[Experiment Error]", entry["experiment_error"], ""]
            if entry.get("critic_feedback"):
                lines += ["[Supervisor Feedback]", entry["critic_feedback"], ""]
        elif action == "final_law":
            lines += ["[Final Law Submitted]", entry.get("final_law", ""), ""]
        elif action in ("warning", "no_tag"):
            lines += [f"[{action.upper()}]", entry.get("system_message", ""), ""]

        # MSE-fit can co-occur with any action (typically alongside an
        # experiment), so emit it in addition to the action-specific block.
        if entry.get("mse_fit_input") is not None:
            lines += ["[MSE Fit Input (law source)]", entry["mse_fit_input"], ""]
        if entry.get("mse_fit_output") is not None:
            lines += [
                "[MSE Fit Output]",
                json.dumps(entry["mse_fit_output"], indent=2),
                "",
            ]

    lines += [sep, "DISCOVERED LAW", sep, law_source or "(none)", ""]

    lines += [sep, "EVALUATION", sep]
    if evaluation:
        for case in evaluation.get("per_case", []):
            lines.append(f"  Case MSE: {case:.4f}")
        lines += [
            f"  Mean particle MSE : {evaluation.get('mean_pos_error', float('inf')):.4f}",
            f"  Max  particle MSE : {evaluation.get('max_pos_error',  float('inf')):.4f}",
            f"  Result            : {'PASS' if evaluation.get('passed') else 'FAIL'}",
        ]
        fit = evaluation.get("fit")
        if fit:
            lines += ["", "[Parameter Fit]"]
            if fit.get("error"):
                lines.append(f"  Error               : {fit['error']}")
            if fit.get("declared_params") is not None:
                lines.append(
                    f"  Declared parameters : {json.dumps(fit['declared_params'])}"
                )
            if fit.get("fitted_params") is not None:
                lines.append(
                    f"  Fitted  parameters  : {json.dumps(fit['fitted_params'])}"
                )
            if fit.get("loss_before") is not None and fit.get("loss_after") is not None:
                lines.append(
                    f"  Training-set loss   : {fit['loss_before']:.4g} -> {fit['loss_after']:.4g}"
                )
        expl = evaluation.get("explanation")
        if expl:
            lines += ["", "[Explanation Metric]"]
            lines.append(
                f"  Agent explanation   : {expl.get('agent_explanation') or '(none submitted)'}"
            )
            lines.append(
                f"  Optimal explanation : {expl.get('optimal_explanation') or '(none defined)'}"
            )
            score = expl.get("score")
            raw = expl.get("raw_score")
            if score is None:
                lines.append(
                    f"  Explanation score   : N/A  ({expl.get('error') or 'unknown error'})"
                )
            else:
                lines.append(f"  Explanation score   : {score:.2f}  (raw {raw}/10)")
            if expl.get("reasoning"):
                lines += ["  Judge reasoning     :", expl["reasoning"].strip()]
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# plotting code to be refactored and located elsewhere in future
def _plot_trajectories(trajectories, world, model, path):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from matplotlib.gridspec import GridSpec

    mpl.rcParams["font.family"] = "serif"

    n = len(trajectories)
    fig = plt.figure(figsize=(4.5 * n, 6))
    gs = GridSpec(
        1, n, figure=fig, wspace=0.45, left=0.05, right=0.97, top=0.88, bottom=0.12
    )

    for j, (ax, traj) in enumerate(
        zip([fig.add_subplot(gs[0, j]) for j in range(n)], trajectories)
    ):
        gt = np.asarray(traj["gt"])
        ax.plot(gt[:, 0], gt[:, 1], "o-", color="black", label="Ground truth", zorder=3)
        if traj["pred"] is not None:
            pred = np.asarray(traj["pred"])
            ax.plot(
                pred[:, 0],
                pred[:, 1],
                "s--",
                color="darkorange",
                label="Predicted",
                zorder=4,
            )
        ax.set_title(
            f"Case {traj['case']}  (p1={traj['p1']}, p2={traj['p2']})\nerr={traj['error']:.4f}",
            fontsize=9,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

    model_short = os.path.basename(model)
    fig.suptitle(f"World: {world}  |  Model: {model_short}", fontsize=11, y=0.98)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def _plot_circle_trajectories(trajectories, world, model, path):
    """Plot ring particle trajectories (GT then pred on top) for the circle world."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    mpl.rcParams["font.family"] = "serif"

    traj = trajectories[0]
    gt = np.asarray(traj["gt"])
    pred = np.asarray(traj["pred"]) if traj["pred"] is not None else None

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.1)

    for i in range(1, gt.shape[1]):
        ax.plot(
            gt[:, i, 0],
            gt[:, i, 1],
            "-",
            color="black",
            lw=1.2,
            alpha=0.8,
            label="Ground truth" if i == 1 else None,
            zorder=3,
        )
    ax.scatter(
        gt[-1, 1:, 0],
        gt[-1, 1:, 1],
        color="black",
        s=40,
        zorder=5,
        marker="o",
        edgecolors="none",
    )

    if pred is not None:
        for i in range(1, pred.shape[1]):
            ax.plot(
                pred[:, i, 0],
                pred[:, i, 1],
                "--",
                color="tomato",
                lw=1.2,
                alpha=0.9,
                label="Predicted" if i == 1 else None,
                zorder=4,
            )
        ax.scatter(
            pred[-1, 1:, 0],
            pred[-1, 1:, 1],
            color="#E65C00",
            s=60,
            zorder=6,
            marker="x",
            linewidths=1.5,
        )

    r, v, err = traj.get("ring_radius", "?"), traj.get("v_tang", "?"), traj["error"]
    ax.set_title(
        f"Circle world  (r={r}, v_tang={v})\nmean err = {err:.4f}", fontsize=10
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.minorticks_on()

    model_short = os.path.basename(model)
    fig.suptitle(f"World: {world}  |  Model: {model_short}", fontsize=11, y=0.98)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


# doesn't seems to work check later
def _plot_species_trajectories(trajectories, world, model, path):
    """Plot full-state multi-particle worlds (N-aware): ground truth for
    *every* particle vs. predicted. Worlds with a genuine two-species A/B
    split (``species``, ``non_symmetric``) are coloured blue/green by
    first-half / second-half; ``gravity_hard`` (hidden mass groups) and
    ``coulomb_hard`` (agent-set signed charges) use a single GT colour."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    mpl.rcParams["font.family"] = "serif"

    n_cases = len(trajectories)
    fig, axes = plt.subplots(1, n_cases, figsize=(5.5 * n_cases, 5.5), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, wspace=0.35)

    color_a = "#2166ac"  # blue for species A (and single-population GT)
    color_b = "#1a9850"  # green for species B
    color_pred = "tomato"
    # Genuine two-species A/B worlds split first-half vs second-half;
    # others (gravity_hard mass groups, coulomb_hard signed charges) are
    # one GT population drawn in a single colour.
    two_species = world in ("species", "non_symmetric")

    for idx, traj in enumerate(trajectories):
        ax = axes[0, idx]
        gt = np.asarray(traj["gt"])  # (T, N, 2)
        pred = np.asarray(traj["pred"]) if traj["pred"] is not None else None
        n = gt.shape[1]
        # species_a is a set so the marker loops below colour correctly
        # whether or not this is a two-species world.
        species_a = set(range(0, n // 2)) if two_species else set(range(n))

        # Ground-truth trajectories for ALL particles, coloured by species
        labelled: set[str] = set()
        for i in range(n):
            if two_species:
                key = "A" if i in species_a else "B"
                lbl = (
                    ("Species A (GT)" if key == "A" else "Species B (GT)")
                    if key not in labelled
                    else None
                )
            else:
                key = "GT"
                lbl = "Ground truth" if key not in labelled else None
            labelled.add(key)
            ax.plot(
                gt[:, i, 0],
                gt[:, i, 1],
                "-",
                color=color_a if i in species_a else color_b,
                lw=1.2,
                alpha=0.8,
                label=lbl,
                zorder=3,
            )

        # Mark initial positions with particle index labels
        for i in range(n):
            c = color_a if i in species_a else color_b
            ax.scatter(
                gt[0, i, 0],
                gt[0, i, 1],
                color=c,
                s=50,
                zorder=7,
                marker="o",
                edgecolors="white",
                linewidths=0.5,
            )
            ax.annotate(
                str(i),
                (gt[0, i, 0], gt[0, i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color=c,
                fontweight="bold",
                zorder=8,
            )

        # Mark final GT positions
        for i in range(n):
            c = color_a if i in species_a else color_b
            ax.scatter(
                gt[-1, i, 0],
                gt[-1, i, 1],
                color=c,
                s=40,
                zorder=6,
                marker="D",
                edgecolors="white",
                linewidths=0.5,
            )

        # Predicted trajectories (single color)
        if pred is not None:
            for i in range(pred.shape[1]):
                ax.plot(
                    pred[:, i, 0],
                    pred[:, i, 1],
                    "--",
                    color=color_pred,
                    lw=1.0,
                    alpha=0.85,
                    label="Predicted" if i == 0 else None,
                    zorder=4,
                )
            ax.scatter(
                pred[-1, :, 0],
                pred[-1, :, 1],
                color="#E65C00",
                s=50,
                zorder=6,
                marker="x",
                linewidths=1.5,
            )

        ax.set_title(
            f"Case {traj['case']}  |  mean err = {traj['error']:.4f}", fontsize=10
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8, loc="best")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

    model_short = os.path.basename(model)
    fig.suptitle(f"World: {world}  |  Model: {model_short}", fontsize=11, y=0.98)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def _plot_dark_matter_trajectories(trajectories, world, model, path):
    """Plot dark matter world: probe trajectories (true vs predicted) + field panel.

    Top row:  Trajectory view — visible (faint blue context), dark matter initial
              positions (black), true probe trajectories (gold), predicted probe
              trajectories (orchid dashes). Error is computed on probes only.
    Bottom:   Final field phi + final positions of visible (blue), dark matter
              (black), true probes (gold), predicted probes (green).
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    mpl.rcParams["font.family"] = "serif"

    n_cases = len(trajectories)
    fig, axes = plt.subplots(2, n_cases, figsize=(7 * n_cases, 13), squeeze=False)
    fig.subplots_adjust(
        left=0.06, right=0.95, top=0.93, bottom=0.04, wspace=0.30, hspace=0.22
    )

    color_vis = "#2166ac"  # blue  — visible background
    color_probe = "#e6ab02"  # gold  — probes (ground truth)
    color_dark = "black"  # black — dark matter (revealed)
    color_pred = "orchid"  # orchid — predicted probes

    # Simulation-internal indices (35-particle full output)
    vis_sim = list(range(0, 20))
    dark_sim = list(range(20, 30))
    probe_sim = list(range(30, 35))

    # Agent-facing indices in the 25-particle output / prediction
    probe_agent = list(range(20, 25))

    def _draw_arrows(ax, traj_data, indices, color, size, zorder):
        """Draw arrowheads at trajectory endpoints."""
        for i in indices:
            if traj_data.shape[0] < 2:
                continue
            end = traj_data[-1, i]
            prev = traj_data[-2, i]
            d = end - prev
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                continue
            d = d / norm
            arrow_len = 0.8
            tail = end - arrow_len * d
            ax.annotate(
                "",
                xy=end,
                xytext=tail,
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=1.5, mutation_scale=size
                ),
                zorder=zorder,
            )

    for idx, traj in enumerate(trajectories):
        # ── Top panel: trajectory plot ──
        ax = axes[0, idx]
        gt_full = np.asarray(traj["gt_full"])  # (T, 35, 2)
        pred = np.asarray(traj["pred"]) if traj["pred"] is not None else None

        # Visible background trajectories
        for i in vis_sim:
            ax.plot(
                gt_full[:, i, 0],
                gt_full[:, i, 1],
                "-",
                color=color_vis,
                lw=0.6,
                alpha=0.4,
                label="Visible (s=1)" if i == vis_sim[0] else None,
                zorder=2,
            )
        # Visible start markers
        ax.scatter(
            gt_full[0, vis_sim, 0],
            gt_full[0, vis_sim, 1],
            color=color_vis,
            s=20,
            zorder=6,
            marker="o",
            edgecolors="white",
            linewidths=0.3,
        )
        _draw_arrows(ax, gt_full, vis_sim, color_vis, 10, 6)

        # Dark matter: initial position markers only
        ax.scatter(
            gt_full[0, dark_sim, 0],
            gt_full[0, dark_sim, 1],
            color=color_dark,
            s=80,
            zorder=8,
            marker="o",
            edgecolors="white",
            linewidths=1.0,
            label="Dark matter (s=5)",
        )

        # True probe trajectories (gold, prominent)
        for j, i in enumerate(probe_sim):
            ax.plot(
                gt_full[:, i, 0],
                gt_full[:, i, 1],
                "-",
                color=color_probe,
                lw=1.8,
                alpha=0.9,
                label="True probes" if j == 0 else None,
                zorder=5,
            )
        # Start markers for probes
        ax.scatter(
            gt_full[0, probe_sim, 0],
            gt_full[0, probe_sim, 1],
            color=color_probe,
            s=40,
            zorder=9,
            marker="D",
            edgecolors="white",
            linewidths=0.5,
        )
        _draw_arrows(ax, gt_full, probe_sim, color_probe, 14, 7)

        # Predicted probe trajectories (orchid dashes)
        if pred is not None:
            pred = np.asarray(pred)  # (T, 25, 2)
            for j, i in enumerate(probe_agent):
                ax.plot(
                    pred[:, i, 0],
                    pred[:, i, 1],
                    "--",
                    color=color_pred,
                    lw=1.4,
                    alpha=0.8,
                    label="Predicted probes" if j == 0 else None,
                    zorder=4,
                )
            if pred.shape[0] >= 2:
                _draw_arrows(ax, pred, probe_agent, color_pred, 12, 6)

        ax.set_title(
            f"Case {traj['case']} probe trajectories  |  mean err = {traj['error']:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

        # ── Bottom panel: field at final time + final positions ──
        ax2 = axes[1, idx]
        field_snaps = traj.get("field_snapshots")
        if field_snaps and len(field_snaps) > 0:
            field = np.asarray(field_snaps[-1])
            domain = 50.0
            centre = domain / 2.0
            extent = [-centre, centre, -centre, centre]
            im = ax2.imshow(
                field.T, origin="lower", extent=extent, cmap="RdBu_r", aspect="equal"
            )
            fig.colorbar(im, ax=ax2, shrink=0.7, label=r"$\varphi$")

            # Final dark matter positions
            ax2.scatter(
                gt_full[-1, dark_sim, 0],
                gt_full[-1, dark_sim, 1],
                color=color_dark,
                s=90,
                zorder=8,
                marker="o",
                edgecolors="white",
                linewidths=1.2,
                label="Dark matter",
            )

            # Final true visible positions (context)
            ax2.scatter(
                gt_full[-1, vis_sim, 0],
                gt_full[-1, vis_sim, 1],
                color=color_vis,
                s=20,
                zorder=6,
                marker="o",
                edgecolors="white",
                linewidths=0.4,
                alpha=0.5,
                label="Visible",
            )

            # Final true probe positions
            ax2.scatter(
                gt_full[-1, probe_sim, 0],
                gt_full[-1, probe_sim, 1],
                color=color_probe,
                s=50,
                zorder=9,
                marker="D",
                edgecolors="white",
                linewidths=0.6,
                label="True probes",
            )

            # Final predicted probe positions
            if pred is not None:
                pred_arr = np.asarray(pred)
                pred_probes_final = pred_arr[-1, probe_agent]
                ax2.scatter(
                    pred_probes_final[:, 0],
                    pred_probes_final[:, 1],
                    color="#2ca02c",
                    s=50,
                    zorder=10,
                    marker="D",
                    edgecolors="white",
                    linewidths=0.6,
                    label="Predicted probes",
                )

            ax2.set_title(
                f"Case {traj['case']} final config "
                + r"$\varphi$"
                + f" at t={traj['times'][-1]:.1f}",
                fontsize=10,
            )
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.legend(fontsize=7, loc="best")
        else:
            ax2.text(
                0.5,
                0.5,
                "No field data",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )
            ax2.set_title(f"Case {traj['case']} field", fontsize=10)

        ax2.tick_params(direction="in", which="both", top=True, right=True)
        ax2.minorticks_on()

    model_short = os.path.basename(model)
    fig.suptitle(f"World: {world}  |  Model: {model_short}", fontsize=12, y=0.97)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def _plot_three_species_trajectories(trajectories, world, model, path):
    """Plot 35-particle three-species world: A (blue), B (green), C (red), probes (gold)."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    mpl.rcParams["font.family"] = "serif"

    n_cases = len(trajectories)
    fig, axes = plt.subplots(1, n_cases, figsize=(7 * n_cases, 7), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.88, bottom=0.06, wspace=0.30)

    color_a = "#2166ac"  # blue  — species A (source=+1)
    color_b = "#1a9850"  # green — species B (source=+3)
    color_c = "#d73027"  # red   — species C (source=-2)
    color_probe = "#e6ab02"  # gold  — probes (source=0)
    color_pred = "orchid"

    species_a = list(range(0, 10))
    species_b = list(range(10, 20))
    species_c = list(range(20, 30))
    probes = list(range(30, 35))

    for idx, traj in enumerate(trajectories):
        ax = axes[0, idx]
        gt = np.asarray(traj["gt"])  # (T, 35, 2)
        pred = np.asarray(traj["pred"]) if traj["pred"] is not None else None

        # Ground truth trajectories by species
        for i in species_a:
            ax.plot(
                gt[:, i, 0],
                gt[:, i, 1],
                "-",
                color=color_a,
                lw=0.6,
                alpha=0.5,
                label="Species A (s=1)" if i == species_a[0] else None,
                zorder=2,
            )
        for i in species_b:
            ax.plot(
                gt[:, i, 0],
                gt[:, i, 1],
                "-",
                color=color_b,
                lw=0.6,
                alpha=0.5,
                label="Species B (s=3)" if i == species_b[0] else None,
                zorder=2,
            )
        for i in species_c:
            ax.plot(
                gt[:, i, 0],
                gt[:, i, 1],
                "-",
                color=color_c,
                lw=0.6,
                alpha=0.5,
                label="Species C (s=-2)" if i == species_c[0] else None,
                zorder=2,
            )
        for i in probes:
            ax.plot(
                gt[:, i, 0],
                gt[:, i, 1],
                "-",
                color=color_probe,
                lw=1.4,
                alpha=0.9,
                label="Probes (s=0)" if i == probes[0] else None,
                zorder=3,
            )

        # Mark initial positions
        for grp, c, m in [
            (species_a, color_a, "o"),
            (species_b, color_b, "s"),
            (species_c, color_c, "^"),
            (probes, color_probe, "D"),
        ]:
            ax.scatter(
                gt[0, grp, 0],
                gt[0, grp, 1],
                color=c,
                s=25,
                zorder=7,
                marker=m,
                edgecolors="white",
                linewidths=0.4,
            )

        # Mark final GT positions
        for grp, c, m in [
            (species_a, color_a, "o"),
            (species_b, color_b, "s"),
            (species_c, color_c, "^"),
            (probes, color_probe, "D"),
        ]:
            ax.scatter(
                gt[-1, grp, 0],
                gt[-1, grp, 1],
                color=c,
                s=35,
                zorder=6,
                marker=m,
                edgecolors="black",
                linewidths=0.5,
            )

        # Predicted trajectories
        if pred is not None:
            pred = np.asarray(pred)
            for i in range(pred.shape[1]):
                ax.plot(
                    pred[:, i, 0],
                    pred[:, i, 1],
                    "--",
                    color=color_pred,
                    lw=0.7,
                    alpha=0.7,
                    label="Predicted" if i == 0 else None,
                    zorder=4,
                )
            ax.scatter(
                pred[-1, :, 0],
                pred[-1, :, 1],
                color=color_pred,
                s=30,
                zorder=5,
                marker="x",
                linewidths=1.0,
            )

        ax.set_title(
            f"Case {traj['case']}  |  mean err = {traj['error']:.4f}", fontsize=10
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

    model_short = os.path.basename(model)
    fig.suptitle(f"World: {world}  |  Model: {model_short}", fontsize=11, y=0.98)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def _plot_ether_trajectories(trajectories, world, model, path):
    """Plot the 26-particle ether world: anchor, mass-coloured orbiters, probes (true vs predicted)."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    mpl.rcParams["font.family"] = "serif"

    n_cases = len(trajectories)
    fig, axes = plt.subplots(1, n_cases, figsize=(7 * n_cases, 7), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.88, bottom=0.06, wspace=0.30)

    color_anchor = "black"
    color_m1 = "#2166ac"  # blue  — orbiters with mass 1
    color_m2 = "#1a9850"  # green — orbiters with mass 2
    color_m4 = "#d73027"  # red   — orbiters with mass 4
    color_probe = "#e6ab02"  # gold — probes (ground truth)
    color_pred = "orchid"

    anchor_idx = [0]
    ring_idx = list(range(1, 21))
    probe_idx = list(range(21, 26))

    # Mass pattern matches NBodyEtherExecutor.MASS_PATTERN cycled across the 20 orbiters
    mass_pattern = (1.0, 2.0, 4.0)
    color_by_mass = {1.0: color_m1, 2.0: color_m2, 4.0: color_m4}
    ring_colors = [
        color_by_mass[mass_pattern[i % len(mass_pattern)]] for i in range(len(ring_idx))
    ]

    for idx, traj in enumerate(trajectories):
        ax = axes[0, idx]
        gt = np.asarray(traj["gt"])  # (T, 26, 2)
        pred = np.asarray(traj["pred"]) if traj["pred"] is not None else None

        # Anchor
        ax.plot(
            gt[:, 0, 0],
            gt[:, 0, 1],
            "-",
            color=color_anchor,
            lw=1.5,
            alpha=0.9,
            zorder=5,
            label="Anchor",
        )
        ax.scatter(
            gt[0, 0, 0],
            gt[0, 0, 1],
            color=color_anchor,
            s=80,
            marker="*",
            zorder=8,
            edgecolors="white",
            linewidths=0.6,
        )

        # Ring orbiters, coloured by mass
        seen_legend = {1.0: False, 2.0: False, 4.0: False}
        for k, i in enumerate(ring_idx):
            m = mass_pattern[k % len(mass_pattern)]
            label = f"Orbiter (m={int(m)})" if not seen_legend[m] else None
            seen_legend[m] = True
            ax.plot(
                gt[:, i, 0],
                gt[:, i, 1],
                "-",
                color=ring_colors[k],
                lw=0.8,
                alpha=0.6,
                zorder=3,
                label=label,
            )
        ax.scatter(
            gt[0, ring_idx, 0],
            gt[0, ring_idx, 1],
            color=ring_colors,
            s=25,
            marker="o",
            zorder=6,
            edgecolors="white",
            linewidths=0.4,
        )

        # True probe trajectories
        for k, i in enumerate(probe_idx):
            ax.plot(
                gt[:, i, 0],
                gt[:, i, 1],
                "-",
                color=color_probe,
                lw=1.6,
                alpha=0.95,
                zorder=5,
                label="True probes" if k == 0 else None,
            )
        ax.scatter(
            gt[0, probe_idx, 0],
            gt[0, probe_idx, 1],
            color=color_probe,
            s=45,
            marker="D",
            zorder=9,
            edgecolors="white",
            linewidths=0.5,
        )

        # Predicted probe trajectories
        if pred is not None:
            for k, i in enumerate(probe_idx):
                ax.plot(
                    pred[:, i, 0],
                    pred[:, i, 1],
                    "--",
                    color=color_pred,
                    lw=1.4,
                    alpha=0.85,
                    zorder=4,
                    label="Predicted probes" if k == 0 else None,
                )

        ax.set_title(
            f"Case {traj['case']}  |  mean probe err = {traj['error']:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

    model_short = os.path.basename(model)
    fig.suptitle(f"World: {world}  |  Model: {model_short}", fontsize=11, y=0.98)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def _plot_circle_law(law_source, world, model, path):
    """Render the discovered law as a standalone figure."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")
    ax.text(
        0.02,
        0.5,
        law_source or "(no law submitted)",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="center",
        horizontalalignment="left",
        family="monospace",
        color="#1a3a6b",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#d8d8d8", edgecolor="#aaaaaa"),
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


def _plot_law(law_source, world, model, path):
    """Render the discovered law as a standalone figure (non-circle worlds)."""
    _plot_circle_law(law_source, world, model, path)


if __name__ == "__main__":
    main()
