"""Run the procurement stages in order using the current Python interpreter."""

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = (
    "search_agent.py",
    "collection_agent.py",
    "scraping_agent.py",
    "comparison_agent.py",
    "report_agent.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dry-run", action="store_true", help="Check and list stages without running them.")
    args = parser.parse_args()
    root = args.project_dir.resolve()
    missing = [stage for stage in STAGES if not (root / stage).is_file()]
    if missing:
        parser.error(f"Missing stages in {root}: {', '.join(missing)}")

    for index, stage in enumerate(STAGES, start=1):
        print(f"[{index}/{len(STAGES)}] {stage}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run([sys.executable, str(root / stage)], cwd=root)
        if result.returncode:
            print(f"Pipeline stopped at {stage} (exit {result.returncode}).", file=sys.stderr)
            return result.returncode

    if not args.dry_run:
        print(f"Report: {root / 'reports' / 'procurement_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
