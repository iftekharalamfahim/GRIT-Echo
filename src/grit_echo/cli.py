"""CLI entry points: ``grit-echo run`` and ``grit-echo audit``."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Settings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(prog="grit-echo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run pipeline stages end to end.")
    p_run.add_argument("--stages", default="audit,seg,eval,rq",
                       help="Comma-separated subset of: audit,seg,eval,rq")
    p_run.add_argument("--config", default=None, help="Optional YAML overrides.")
    p_run.add_argument("--force-seg", action="store_true", help="Retrain U-Net.")

    sub.add_parser("audit", help="Cohort audit only (fast, no GPU).")

    args = parser.parse_args(argv)

    cfg = Settings.load(args.config if hasattr(args, "config") else None)

    if args.command == "audit":
        from .data import cohort_audit_table
        table = cohort_audit_table(cfg)
        print(f"{table.patient.nunique()} patients, {len(table)} frames")
        return 0

    from .pipeline import run
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    run(cfg, stages=stages, force_seg=args.force_seg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
