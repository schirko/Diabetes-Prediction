"""Re-run every notebook in notebooks/ from top to bottom and save the results.

    python -m src.run_notebooks            all notebooks
    python -m src.run_notebooks 02         only those whose filename contains "02"

Run this after changing anything in src/ or the data, so the saved outputs
never drift from the code. It stops at the first notebook that raises an error.
"""
import sys
import time

import nbformat
from nbclient import NotebookClient

from .data import ROOT

NB_DIR = ROOT / "notebooks"


def run(pattern: str = "") -> None:
    paths = sorted(p for p in NB_DIR.glob("*.ipynb") if pattern in p.name)
    if not paths:
        sys.exit(f"No notebooks match '{pattern}' in {NB_DIR}")
    for path in paths:
        start = time.time()
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
        nbformat.write(nb, path)
        print(f"ran {path.name}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "")
