#!/usr/bin/env python3
"""Compatibility wrapper for running Phase 1 without the console-script name."""

from cot_faithfulness.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
