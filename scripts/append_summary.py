#!/usr/bin/env python
"""Append one summary row (JSONL) extracted from a run_discovery.py output file."""

import argparse
import json
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-json", required=True, help="Path to <base>.json from --store_output"
    )
    p.add_argument(
        "--summary", required=True, help="Path to summary.jsonl to append to"
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--critic", required=True, choices=["on", "off"])
    p.add_argument("--critic-model", default="")
    p.add_argument("--return-code", type=int, default=0)
    p.add_argument("--stdout-log", default="")
    p.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Round budget configured for the run (for rounds sweeps).",
    )
    args = p.parse_args()

    use_critic = args.critic == "on"

    if not os.path.exists(args.run_json):
        row = {
            "model": None,
            "world": None,
            "noise_std": None,
            "noise_seed": args.seed,
            "seed": args.seed,
            "use_critic": use_critic,
            "critic_model": args.critic_model if use_critic else None,
            "max_rounds": args.max_rounds,
            "mean_pos_error": None,
            "max_pos_error": None,
            "passed": False,
            "per_case_errors": None,
            "n_rounds": None,
            "explanation_raw_score": None,
            "explanation_score": None,
            "final_law": None,
            "run_json_path": os.path.abspath(args.run_json),
            "stdout_log_path": (
                os.path.abspath(args.stdout_log) if args.stdout_log else None
            ),
            "return_code": args.return_code,
            "status": "missing_output",
        }
    else:
        with open(args.run_json) as f:
            data = json.load(f)

        ev = data.get("evaluation") or {}
        expl = ev.get("explanation") or {}

        row = {
            "timestamp": data.get("timestamp"),
            "model": data.get("model"),
            "world": data.get("world"),
            "noise_std": data.get("noise_std"),
            "noise_seed": data.get("noise_seed"),
            "seed": args.seed,
            "use_critic": use_critic,
            "critic_model": args.critic_model if use_critic else None,
            "max_rounds": data.get("max_rounds", args.max_rounds),
            "mean_pos_error": ev.get("mean_pos_error"),
            "max_pos_error": ev.get("max_pos_error"),
            "passed": bool(ev.get("passed", False)),
            "per_case_errors": ev.get("per_case"),
            "n_rounds": len(data.get("rounds") or []),
            "explanation_raw_score": expl.get("raw_score"),
            "explanation_score": expl.get("score"),
            "final_law": data.get("final_law"),
            "run_json_path": os.path.abspath(args.run_json),
            "stdout_log_path": (
                os.path.abspath(args.stdout_log) if args.stdout_log else None
            ),
            "return_code": args.return_code,
            "status": "ok" if args.return_code == 0 else "nonzero_exit",
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[append_summary] error: {e}", file=sys.stderr)
        sys.exit(1)
