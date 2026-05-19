#!/usr/bin/env python
"""
Aggregate a benchmark summary.jsonl into:
  - summary_agg.csv         one row per (model, world, noise, critic)
  - summary_table.md        the same, formatted as a markdown table
  - mse_bar.png             grouped bar chart of mean position error
  - explanation_bar.png     grouped bar chart of explanation score

Invariant values (mean ± std) are computed across seeds. Failed runs with
inf/None mean_pos_error are excluded from the mean but counted toward the
pass rate denominator.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, stdev


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _clean(values):
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        out.append(f)
    return out


def agg(values):
    vals = _clean(values)
    if not vals:
        return None, None, 0
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0
    return m, s, len(vals)


def geomean_agg(values):
    """Geometric mean with asymmetric (multiplicative) error offsets.
    Returns (gmean, (lower_offset, upper_offset), n). Offsets are additive
    distances from the mean to mean/gsd and mean*gsd, suitable for
    matplotlib yerr on a log-scale axis. Non-positive values are dropped
    since log is undefined there."""
    vals = [v for v in _clean(values) if v > 0]
    if not vals:
        return None, None, 0
    logs = [math.log(v) for v in vals]
    lm = mean(logs)
    gm = math.exp(lm)
    if len(logs) > 1:
        ls = stdev(logs)
        gsd = math.exp(ls)
        err = (gm - gm / gsd, gm * gsd - gm)
    else:
        err = (0.0, 0.0)
    return gm, err, len(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    rows = load(args.summary)
    if not rows:
        print(f"[aggregate_bench] no rows in {args.summary}")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    # Preserve world ordering as seen in the summary (matches difficulty ladder).
    world_order = []
    seen = set()
    for r in rows:
        w = r.get("world")
        if w and w not in seen:
            seen.add(w)
            world_order.append(w)

    models = sorted({r["model"] for r in rows if r.get("model")})
    noises = sorted({r["noise_std"] for r in rows if r.get("noise_std") is not None})

    # Group by (model, world, noise_std, use_critic)
    groups = defaultdict(list)
    for r in rows:
        if r.get("model") is None or r.get("world") is None:
            continue
        key = (r["model"], r["world"], r["noise_std"], bool(r["use_critic"]))
        groups[key].append(r)

    # ---------- CSV + Markdown table ----------
    csv_path = os.path.join(args.out_dir, "summary_agg.csv")
    md_path = os.path.join(args.out_dir, "summary_table.md")

    md_lines = [
        "| model | world | noise | critic | n | MSE (mean±std) | explanation (mean±std) | pass rate | rounds (mean) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    csv_rows = [
        [
            "model",
            "world",
            "noise_std",
            "use_critic",
            "n",
            "mean_pos_error_mean",
            "mean_pos_error_std",
            "explanation_score_mean",
            "explanation_score_std",
            "pass_rate",
            "n_rounds_mean",
        ]
    ]

    def sort_key(k):
        model, world, noise, critic = k
        try:
            w_idx = world_order.index(world)
        except ValueError:
            w_idx = len(world_order)
        return (model, w_idx, noise, critic)

    for key in sorted(groups.keys(), key=sort_key):
        model, world, noise, critic = key
        grp = groups[key]
        mse_m, mse_s, mse_n = agg([r["mean_pos_error"] for r in grp])
        exp_m, exp_s, exp_n = agg([r["explanation_score"] for r in grp])
        rounds_m, _, _ = agg([r["n_rounds"] for r in grp])
        passed_vals = [1.0 if r.get("passed") else 0.0 for r in grp]
        pass_rate = mean(passed_vals) if passed_vals else 0.0

        mse_str = f"{mse_m:.3f}±{mse_s:.3f}" if mse_m is not None else "—"
        exp_str = f"{exp_m:.2f}±{exp_s:.2f}" if exp_m is not None else "—"
        rounds_str = f"{rounds_m:.1f}" if rounds_m is not None else "—"

        md_lines.append(
            f"| {model} | {world} | {noise} | {'on' if critic else 'off'} | "
            f"{len(grp)} | {mse_str} | {exp_str} | {pass_rate:.2f} | {rounds_str} |"
        )
        csv_rows.append(
            [
                model,
                world,
                noise,
                "on" if critic else "off",
                len(grp),
                mse_m,
                mse_s,
                exp_m,
                exp_s,
                pass_rate,
                rounds_m,
            ]
        )

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")

    # ---------- Plots ----------
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    x = np.arange(len(world_order))
    n_bars = len(models) * len(noises)
    width = 0.8 / max(n_bars, 1)
    # Hand-picked pastel palette (dusty teal, sage, dusty rose, peach, powder blue, butter, mauve, mint).
    pastel_palette = [
        "#8FB5C7",
        "#A8D5BA",
        "#F4B6C2",
        "#F5CBA7",
        "#AECDE0",
        "#E8D5A0",
        "#D4A5A5",
        "#C5E1C5",
    ]
    # Muted jewel-tone palette for dark backgrounds — readable without being neon.
    cyberpunk_palette = [
        "#6FA8C7",
        "#C77D8E",
        "#9A8CC2",
        "#D4B26A",
        "#7FB3A0",
        "#C79470",
        "#A3B8CC",
        "#B89BC0",
    ]

    def _draw_panel(
        ax,
        metric,
        ylabel,
        critic_on,
        logy=False,
        show_legend=False,
        tick_fs=9,
        label_fs=None,
        title_fs=None,
        legend_fs=9,
        lowercase=False,
        edgecolor="black",
        ecolor=None,
        palette=pastel_palette,
        diff=False,
        invert_diff=False,
        log_ratio=False,
        show_helped_annotation=False,
    ):
        # Use geometric mean for MSE panels so the bar height matches the
        # log y-axis visual intuition (and isn't dominated by one outlier).
        use_geomean = (metric == "mean_pos_error") and not diff
        for mi, model in enumerate(models):
            for ni, noise in enumerate(noises):
                means, errs_lo, errs_hi = [], [], []
                for world in world_order:
                    if diff:
                        on_vals = {
                            r.get("seed"): r[metric]
                            for r in rows
                            if r.get("model") == model
                            and r.get("world") == world
                            and r.get("noise_std") == noise
                            and bool(r.get("use_critic"))
                        }
                        off_vals = {
                            r.get("seed"): r[metric]
                            for r in rows
                            if r.get("model") == model
                            and r.get("world") == world
                            and r.get("noise_std") == noise
                            and not bool(r.get("use_critic"))
                        }
                        diffs = []
                        for sd_key in set(on_vals) & set(off_vals):
                            try:
                                ov = float(on_vals[sd_key])
                                fv = float(off_vals[sd_key])
                            except (TypeError, ValueError):
                                continue
                            if not (math.isfinite(ov) and math.isfinite(fv)):
                                continue
                            if log_ratio:
                                if ov <= 0 or fv <= 0:
                                    continue
                                diffs.append(
                                    math.log10(fv / ov)
                                    if invert_diff
                                    else math.log10(ov / fv)
                                )
                            else:
                                diffs.append(fv - ov if invert_diff else ov - fv)
                        if diffs:
                            m = sum(diffs) / len(diffs)
                            s = stdev(diffs) if len(diffs) > 1 else 0.0
                        else:
                            m, s = None, None
                        lo = hi = s if s is not None else 0.0
                    else:
                        g = [
                            r
                            for r in rows
                            if r.get("model") == model
                            and r.get("world") == world
                            and r.get("noise_std") == noise
                            and bool(r.get("use_critic")) == critic_on
                        ]
                        if use_geomean:
                            m, err, _ = geomean_agg([r[metric] for r in g])
                            lo, hi = err if err is not None else (0.0, 0.0)
                        else:
                            m, s, _ = agg([r[metric] for r in g])
                            lo = hi = s if s is not None else 0.0
                    means.append(m if m is not None else 0.0)
                    errs_lo.append(lo)
                    errs_hi.append(hi)
                bar_idx = mi * len(noises) + ni
                offset = (bar_idx - (n_bars - 1) / 2) * width
                color = palette[mi % len(palette)]
                hatch = "" if noise == noises[0] else "//"
                label = f"{model}  σ={noise}"
                if lowercase:
                    label = label.lower()
                bar_kwargs = dict(
                    color=color,
                    alpha=0.55 if noise != noises[0] else 0.95,
                    hatch=hatch,
                    edgecolor=edgecolor,
                    linewidth=0.4,
                    capsize=2,
                )
                if ecolor is not None:
                    bar_kwargs["ecolor"] = ecolor
                yerr = np.array([errs_lo, errs_hi])
                ax.bar(x + offset, means, width, yerr=yerr, label=label, **bar_kwargs)
        ax.set_xticks(x)
        world_display = {"dark_matter": "dark matter", "three_species": "multi species"}
        xticklabels = [world_display.get(w, w.replace("_", " ")) for w in world_order]
        if lowercase:
            xticklabels = [t.lower() for t in xticklabels]
        ax.set_xticklabels(xticklabels, rotation=20, ha="right", fontsize=tick_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)
        if diff and log_ratio:
            ylabel_used = "log₁₀(mse off / on)" if lowercase else "log₁₀(MSE off / on)"
        elif diff:
            ylabel_used = "residual (critic)"
        elif use_geomean:
            ylabel_used = (
                ylabel.lower().replace("mse", "mse (geomean)")
                if lowercase
                else ylabel.replace("MSE", "MSE (geomean)")
            )
        else:
            ylabel_used = ylabel.lower() if lowercase else ylabel
        ax.set_ylabel(ylabel_used, fontsize=label_fs)
        if diff:
            title = "critic effect" if lowercase else "Critic effect"
        else:
            title = (
                f"critic {'on' if critic_on else 'off'}"
                if lowercase
                else f"Critic {'on' if critic_on else 'off'}"
            )
        ax.set_title(title, fontsize=title_fs)
        ax.grid(False)
        if diff:
            ax.axhline(0, color=edgecolor, linewidth=0.8, alpha=0.6)
            if logy and not log_ratio:
                ax.set_yscale("symlog", linthresh=1.0)
            if show_helped_annotation:
                anno_fs = label_fs or 14
                ax.text(
                    0.98,
                    0.96,
                    "↑ critic helped",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=anno_fs,
                    style="italic",
                    color=edgecolor,
                    alpha=0.85,
                )
                ax.text(
                    0.98,
                    0.04,
                    "↓ critic harmed",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=anno_fs,
                    style="italic",
                    color=edgecolor,
                    alpha=0.85,
                )
        elif logy:
            ax.set_yscale("log")
        if show_legend:
            ax.legend(fontsize=legend_fs, loc="best")

    def _grouped_bar(metric, ylabel, fname, logy=False):
        fig, axes = plt.subplots(1, 2, figsize=(max(7, 2.6 * len(world_order)) * 2, 5))
        for ax, critic_on in zip(axes, [False, True]):
            _draw_panel(
                ax,
                metric,
                ylabel,
                critic_on,
                logy=logy,
                show_legend=(critic_on is False),
            )
        fig.suptitle(f"{ylabel} across worlds (mean ± std over seeds)", y=1.02)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")

    def _combined_2x2(fname, dark=False):
        width_in = max(7, 2.6 * len(world_order)) * 3
        style = "dark_background" if dark else "default"
        edgecolor = "white" if dark else "black"
        palette = cyberpunk_palette if dark else pastel_palette
        ecolor = "white" if dark else None
        with plt.style.context(style):
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            fig, axes = plt.subplots(2, 3, figsize=(width_in, 9), sharex=True)
            fs = dict(
                tick_fs=18,
                label_fs=22,
                title_fs=22,
                legend_fs=16,
                lowercase=True,
                edgecolor=edgecolor,
                ecolor=ecolor,
                palette=palette,
            )
            _draw_panel(
                axes[0, 0], "explanation_score", "Explanation score (↑)", False, **fs
            )
            _draw_panel(
                axes[0, 1], "explanation_score", "Explanation score (↑)", True, **fs
            )
            _draw_panel(
                axes[0, 2],
                "explanation_score",
                "Explanation score (↑)",
                True,
                diff=True,
                show_helped_annotation=True,
                **fs,
            )
            _draw_panel(axes[1, 0], "mean_pos_error", "MSE (↓)", False, logy=True, **fs)
            _draw_panel(axes[1, 1], "mean_pos_error", "MSE (↓)", True, logy=True, **fs)
            _draw_panel(
                axes[1, 2],
                "mean_pos_error",
                "MSE (↓)",
                True,
                diff=True,
                invert_diff=True,
                log_ratio=True,
                **fs,
            )
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.05),
                ncol=len(handles),
                fontsize=18,
                frameon=False,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.subplots_adjust(bottom=0.16)

            from matplotlib.patches import FancyArrowPatch

            arrow_y = 0.015
            for ax in axes[1, :]:
                bbox = ax.get_position()
                arrow = FancyArrowPatch(
                    (bbox.x0, arrow_y),
                    (bbox.x1, arrow_y),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=25,
                    color=edgecolor,
                    linewidth=2,
                )
                fig.patches.append(arrow)
                fig.text(
                    (bbox.x0 + bbox.x1) / 2,
                    arrow_y,
                    "physics difficulty",
                    ha="center",
                    va="center",
                    color=edgecolor,
                    fontsize=22,
                    style="italic",
                    bbox=dict(facecolor=fig.get_facecolor(), edgecolor="none", pad=4),
                )

            out_path = os.path.join(args.out_dir, fname)
            fig.savefig(
                out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor()
            )
            plt.close(fig)
        print(f"Wrote {out_path}")

    def _combined_3x2(fname, dark=False):
        width_in = max(7, 2.6 * len(world_order)) * 2
        style = "dark_background" if dark else "default"
        edgecolor = "white" if dark else "black"
        palette = cyberpunk_palette if dark else pastel_palette
        ecolor = "white" if dark else None
        with plt.style.context(style):
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            fig, axes = plt.subplots(3, 2, figsize=(width_in, 13), sharex=True)
            fs = dict(
                tick_fs=18,
                label_fs=22,
                title_fs=22,
                legend_fs=16,
                lowercase=True,
                edgecolor=edgecolor,
                ecolor=ecolor,
                palette=palette,
            )
            # Col 0: explanation score; Col 1: MSE. Rows: off, on, diff.
            _draw_panel(
                axes[0, 0], "explanation_score", "Explanation score (↑)", False, **fs
            )
            _draw_panel(axes[0, 1], "mean_pos_error", "MSE (↓)", False, logy=True, **fs)
            _draw_panel(
                axes[1, 0], "explanation_score", "Explanation score (↑)", True, **fs
            )
            _draw_panel(axes[1, 1], "mean_pos_error", "MSE (↓)", True, logy=True, **fs)
            _draw_panel(
                axes[2, 0],
                "explanation_score",
                "Explanation score (↑)",
                True,
                diff=True,
                show_helped_annotation=True,
                **fs,
            )
            _draw_panel(
                axes[2, 1],
                "mean_pos_error",
                "MSE (↓)",
                True,
                diff=True,
                invert_diff=True,
                log_ratio=True,
                **fs,
            )
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.05),
                ncol=len(handles),
                fontsize=18,
                frameon=False,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.subplots_adjust(bottom=0.11)

            from matplotlib.patches import FancyArrowPatch

            arrow_y = 0.012
            for ax in axes[2, :]:
                bbox = ax.get_position()
                arrow = FancyArrowPatch(
                    (bbox.x0, arrow_y),
                    (bbox.x1, arrow_y),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=25,
                    color=edgecolor,
                    linewidth=2,
                )
                fig.patches.append(arrow)
                fig.text(
                    (bbox.x0 + bbox.x1) / 2,
                    arrow_y,
                    "physics difficulty",
                    ha="center",
                    va="center",
                    color=edgecolor,
                    fontsize=22,
                    style="italic",
                    bbox=dict(facecolor=fig.get_facecolor(), edgecolor="none", pad=4),
                )

            out_path = os.path.join(args.out_dir, fname)
            fig.savefig(
                out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor()
            )
            plt.close(fig)
        print(f"Wrote {out_path}")

    def _draw_aggregate_panel(
        ax,
        critic_on,
        edgecolor="black",
        palette=pastel_palette,
        tick_fs=18,
        label_fs=22,
        title_fs=22,
        legend_fs=14,
        lowercase=True,
    ):
        """
        Right panel of the 2+1 layout: per-model bar chart averaging each
        metric across ALL physics trials (worlds, noise levels, seeds) at
        the given critic setting. Firebrick bars = explanation score (left
        y-axis, linear 0–1). Cornflowerblue bars = MSE (right y-axis, log).
        Bars are annotated with their numeric values.
        """
        exp_means, exp_stds = [], []
        mse_means, mse_err_lo, mse_err_hi = [], [], []
        for model in models:
            g = [
                r
                for r in rows
                if r.get("model") == model and bool(r.get("use_critic")) == critic_on
            ]
            em, es, _ = agg([r["explanation_score"] for r in g])
            mm, merr, _ = geomean_agg([r["mean_pos_error"] for r in g])
            exp_means.append(em if em is not None else 0.0)
            exp_stds.append(es if es is not None else 0.0)
            mse_means.append(mm if mm is not None else 0.0)
            lo, hi = merr if merr is not None else (0.0, 0.0)
            mse_err_lo.append(lo)
            mse_err_hi.append(hi)

        color_exp = "firebrick"
        color_mse = "cornflowerblue"
        x_local = np.arange(len(models))
        width = 0.38

        bars_exp = ax.bar(
            x_local - width / 2,
            exp_means,
            width,
            yerr=exp_stds,
            color=color_exp,
            ecolor=edgecolor,
            capsize=3,
            edgecolor=edgecolor,
            linewidth=0.4,
            label=("explanation score (↑)" if lowercase else "Explanation score (↑)"),
        )
        ax.set_ylabel(
            (
                "mean explanation score (↑)"
                if lowercase
                else "Mean explanation score (↑)"
            ),
            color=color_exp,
            fontsize=label_fs,
        )
        ax.tick_params(axis="y", labelcolor=color_exp, labelsize=tick_fs)
        ax.set_ylim(0.0, 1.15)

        ax2 = ax.twinx()
        mse_yerr = np.array([mse_err_lo, mse_err_hi])
        bars_mse = ax2.bar(
            x_local + width / 2,
            mse_means,
            width,
            yerr=mse_yerr,
            color=color_mse,
            ecolor=edgecolor,
            capsize=3,
            edgecolor=edgecolor,
            linewidth=0.4,
            label=("mse (↓, geomean)" if lowercase else "MSE (↓, geomean)"),
        )
        ax2.set_ylabel(
            (
                "geometric mean mse (↓, log)"
                if lowercase
                else "Geometric mean MSE (↓, log)"
            ),
            color=color_mse,
            fontsize=label_fs,
        )
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor=color_mse, labelsize=tick_fs)

        ax.set_xticks(x_local)
        model_labels = [m.replace("claude-", "") for m in models]
        if lowercase:
            model_labels = [m.lower() for m in model_labels]
        ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=tick_fs)
        ax.set_xlabel("")

    def _combined_2plus1(fname, dark=False):
        """
        Three-panel figure: left column stacks the critic-off explanation
        score and MSE bars; right column spans both rows with both metrics
        averaged across all models. Physics-difficulty arrow under every
        panel's x-axis.
        """
        from matplotlib.gridspec import GridSpec

        width_in = max(7, 2.6 * len(world_order)) * 2
        style = "dark_background" if dark else "default"
        edgecolor = "white" if dark else "black"
        palette = cyberpunk_palette if dark else pastel_palette
        ecolor = "white" if dark else None
        with plt.style.context(style):
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            fig = plt.figure(figsize=(width_in, 10))
            gs = GridSpec(
                2,
                2,
                figure=fig,
                width_ratios=[1.0, 1.15],
                hspace=0.55,
                wspace=0.30,
                left=0.07,
                right=0.94,
                top=0.90,
                bottom=0.15,
            )
            ax_tl = fig.add_subplot(gs[0, 0])
            ax_bl = fig.add_subplot(gs[1, 0], sharex=ax_tl)
            ax_r = fig.add_subplot(gs[:, 1])

            fs = dict(
                tick_fs=18,
                label_fs=22,
                title_fs=22,
                legend_fs=16,
                lowercase=True,
                edgecolor=edgecolor,
                ecolor=ecolor,
                palette=palette,
            )
            _draw_panel(
                ax_tl, "explanation_score", "Explanation score (↑)", False, **fs
            )
            _draw_panel(ax_bl, "mean_pos_error", "MSE (↓)", False, logy=True, **fs)
            _draw_aggregate_panel(
                ax_r,
                critic_on=False,
                edgecolor=edgecolor,
                palette=palette,
                tick_fs=18,
                label_fs=22,
                title_fs=22,
                legend_fs=14,
                lowercase=True,
            )

            # Drop the "critic off" titles — the figure's only state is
            # critic off, so the repeated title is clutter.
            ax_tl.set_title("")
            ax_bl.set_title("")

            # Hide top-left x-tick labels (shared with bottom-left); bottom row
            # keeps them so the physics-difficulty arrow labels a visible axis.
            plt.setp(ax_tl.get_xticklabels(), visible=False)
            ax_tl.set_xlabel("")

            handles, labels = ax_tl.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=min(len(handles), 6),
                fontsize=16,
                frameon=False,
            )

            # Physics-difficulty arrow sits only under the left column, since
            # the right panel's x-axis is now models, not worlds.
            from matplotlib.patches import FancyArrowPatch

            arrow_y = 0.02
            for ax in (ax_bl,):
                bbox = ax.get_position()
                arrow = FancyArrowPatch(
                    (bbox.x0, arrow_y),
                    (bbox.x1, arrow_y),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=25,
                    color=edgecolor,
                    linewidth=2,
                )
                fig.patches.append(arrow)
                fig.text(
                    (bbox.x0 + bbox.x1) / 2,
                    arrow_y,
                    "physics difficulty",
                    ha="center",
                    va="center",
                    color=edgecolor,
                    fontsize=22,
                    style="italic",
                    bbox=dict(facecolor=fig.get_facecolor(), edgecolor="none", pad=4),
                )

            out_path = os.path.join(args.out_dir, fname)
            fig.savefig(
                out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor()
            )
            plt.close(fig)
        print(f"Wrote {out_path}")

    _grouped_bar("mean_pos_error", "MSE (↓)", "mse_bar.png", logy=True)
    _grouped_bar("explanation_score", "Explanation score (↑)", "explanation_bar.png")
    _combined_2x2("explanation_mse_2x2.png")
    _combined_2x2("explanation_mse_2x2_dark.png", dark=True)
    _combined_3x2("explanation_mse_3x2.png")
    _combined_3x2("explanation_mse_3x2_dark.png", dark=True)
    _combined_2plus1("explanation_mse_2plus1.png")
    _combined_2plus1("explanation_mse_2plus1_dark.png", dark=True)


if __name__ == "__main__":
    main()
