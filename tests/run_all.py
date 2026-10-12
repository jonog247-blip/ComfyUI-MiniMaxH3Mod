#!/usr/bin/env python
"""Run every RefMod check from one command.

    python tests/run_all.py            # from the pack directory

Python suites run in subprocesses (each needs a clean ComfyUI import), and the
frontend check runs under Node when it is available. Exits non-zero if any
suite fails, so it is usable as a CI step.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def find_comfy() -> Path:
    """The ComfyUI install to import from.

    A normal install puts the pack at ``<ComfyUI>/custom_nodes/<pack>``, so the
    grandparent is the root; a standalone checkout next to a ComfyUI clone
    needs the sibling. ``COMFYUI_ROOT`` overrides both.
    """
    candidates = []
    if os.environ.get("COMFYUI_ROOT"):
        candidates.append(Path(os.environ["COMFYUI_ROOT"]))
    candidates += [ROOT.parents[1], ROOT.parent / "ComfyUI"]
    for path in candidates:
        if (path / "folder_paths.py").is_file() and (path / "comfy").is_dir():
            return path
    raise SystemExit(
        "Could not find a ComfyUI installation. Set COMFYUI_ROOT=/path/to/ComfyUI "
        f"(tried: {', '.join(str(c) for c in candidates)}).")


COMFY = find_comfy()

PY_SUITES = [
    "tests/test_loaders.py",
    "tests/test_audio_resample.py",
    "tests/test_core_refmod.py",
    "tests/test_refmod_pipeline.py",
    "compatibility_regression_test.py",
    "issue_regression_test.py",
]
NODE_SUITES = ["tests/refmod_slots.test.mjs"]


def run(cmd, cwd):
    print(f"\n=== {' '.join(os.path.basename(c) for c in cmd)} (comfy={COMFY}) ===",
          flush=True)
    env = dict(os.environ, PYTHONPATH=str(COMFY))
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def main() -> int:
    failures = []
    for suite in PY_SUITES:
        if run([sys.executable, str(ROOT / suite)], ROOT):
            failures.append(suite)
    for suite in NODE_SUITES:
        if subprocess.call(["node", "--version"], stdout=subprocess.DEVNULL):
            print(f"\n=== skipping {suite}: node not installed ===")
            continue
        if run(["node", "--test", str(ROOT / suite)], ROOT):
            failures.append(suite)
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"All {len(PY_SUITES)} python suite(s) + node check(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
