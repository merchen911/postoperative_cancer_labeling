#!/usr/bin/env python3
"""Top-level runner for the cancer-conclusion classification pipeline.

Runs the numbered stage scripts under ``scripts/`` in order, via subprocess, passing
the chosen ``--target`` through to each stage that supports it.

Examples
--------
    python run_pipeline.py                                # every stage, both targets
    python run_pipeline.py --stage all --target metas     # every stage, metas only
    python run_pipeline.py --stage train --target recur   # just the training stage
    python run_pipeline.py --stage apply --dry-run        # show the command, don't run it

Stage order (a stage depends on the artifacts written by the ones before it):

    preprocess -> split -> train -> apply -> aggregate ->
    performance -> embedding -> captum -> term_stats

Notes
-----
* The 'aggregate' stage (05) fuses BOTH targets into one master workbook, so it is
  intentionally not per-target: --target is never forwarded to it.
* --target both (the default) omits the flag entirely, so each stage script falls back
  to its own "loop over both recur and metas" default.
* scripts/util_param_count.py is a stand-alone diagnostic, not a pipeline stage, so it
  is deliberately not part of any plan here.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Ordered (stage-name -> script-file) mapping. Order is the execution order.
STAGES: list[tuple[str, str]] = [
    ("preprocess",  "01_preprocess.py"),
    ("split",       "02_split.py"),
    ("train",       "03_train.py"),
    ("apply",       "04_apply.py"),
    ("aggregate",   "05_aggregate.py"),
    ("performance", "06_performance.py"),
    ("embedding",   "07_embedding_analysis.py"),
    ("captum",      "08_captum.py"),
    ("term_stats",  "09_term_stats.py"),
]
STAGE_ORDER = [name for name, _ in STAGES]
STAGE_SCRIPT = dict(STAGES)

# Stages whose script has no --target flag (they handle both targets jointly).
NO_TARGET_STAGES = {"aggregate"}


def build_plan(stage: str) -> list[str]:
    """Return the ordered list of stage names to run for the requested --stage."""
    if stage == "all":
        return list(STAGE_ORDER)
    return [stage]


def stage_command(stage: str, target: str) -> list[str]:
    """Build the subprocess argv for a single stage."""
    cmd = [sys.executable, str(SCRIPTS_DIR / STAGE_SCRIPT[stage])]
    if target != "both" and stage not in NO_TARGET_STAGES:
        cmd += ["--target", target]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the cancer-conclusion classification pipeline stages in order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_ORDER + ["all"],
        default="all",
        help="Which stage to run (default: all, in dependency order).",
    )
    parser.add_argument(
        "--target",
        choices=["recur", "metas", "both"],
        default="both",
        help="Target to forward to each stage (default: both).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and the commands without executing them.",
    )
    args = parser.parse_args()

    plan = build_plan(args.stage)

    # --- Print the ordered plan ---------------------------------------------
    print(f"Pipeline plan  (stage={args.stage}, target={args.target}):")
    for i, stage in enumerate(plan, 1):
        script = STAGE_SCRIPT[stage]
        if stage in NO_TARGET_STAGES:
            tgt = "both (joint, --target not applicable)"
        else:
            tgt = args.target
        print(f"  {i}. {stage:<12s} -> scripts/{script:<20s} [target: {tgt}]")
    print()

    # --- Execute -------------------------------------------------------------
    for stage in plan:
        cmd = stage_command(stage, args.target)
        printable = " ".join(cmd)
        print(f">>> {printable}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(
                f"\nStage '{stage}' failed (exit code {result.returncode}); aborting.",
                file=sys.stderr,
            )
            return result.returncode

    print("\nDone." if not args.dry_run else "\nDry run complete (nothing executed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
