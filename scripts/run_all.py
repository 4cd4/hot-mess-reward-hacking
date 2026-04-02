"""
run_all.py — Master script that runs the full experiment in order.

You can also run each phase manually:
    python -m src.phase1_baseline --model deepseek-v3
    python -m src.phase2_hinted --model deepseek-v3
    python -m src.phase3_analysis

This script runs everything sequentially for all models.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_ORDER


def run(cmd):
    """Run a command and check for errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        sys.exit(1)


def main():
    # Make sure results directory exists
    Path("results").mkdir(exist_ok=True)

    # Check that questions exist
    if not Path("data/questions.json").exists():
        print("ERROR: data/questions.json not found!")
        print("Run: python3 scripts/download_questions.py")
        sys.exit(1)

    python = sys.executable
    models = MODEL_ORDER

    for model in models:
        print(f"\n\n{'#'*60}")
        print(f"# MODEL: {model}")
        print(f"{'#'*60}")

        # Phase 1: Baseline
        run(f"{python} -m src.phase1_baseline --model {model}")

        # Phase 2: Hinted
        run(f"{python} -m src.phase2_hinted --model {model} --all")

    # Phase 3: Analysis (all models together)
    run(f"{python} -m src.phase3_analysis")
    run(f"{python} scripts/plot_results.py")

    print("\n\nDONE! Check the results/ directory for output files.")


if __name__ == "__main__":
    main()
