"""Minimal ESMA entry point."""

import runpy
import sys
from pathlib import Path


def main() -> None:
    source_dir = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(source_dir))
    runpy.run_module("esma.agents.protoss_bot", run_name="__main__")


if __name__ == "__main__":
    main()
