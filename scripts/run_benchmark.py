#!/usr/bin/env python
"""CLI for the watermark benchmark. Equivalent to ``python -m wmbench.cli``.

Examples::

    # everything, one GPU job
    python scripts/run_benchmark.py --preset full --model facebook/opt-1.3b --run-name opt13b

    # pipeline check on CPU/GPU in a few minutes
    python scripts/run_benchmark.py --preset smoke

    # SLURM array: one (variant) per task, then a merge job (see scripts/slurm/)
    python scripts/run_benchmark.py --stage generate --variants KGW --run-name opt13b
    python scripts/run_benchmark.py --stage detect_none,quality,report --run-name opt13b
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmbench.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
