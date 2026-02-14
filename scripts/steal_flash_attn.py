#!/usr/bin/env python3
"""Copy pre-compiled flash-attn from an existing venv (e.g. ACE-Step).

Preferred method:
    uv sync                      # flash-attn auto-installed from prebuilt wheels

This script is a fallback for when the prebuilt wheel doesn't match your
Python/torch/CUDA combination and you already have flash-attn compiled
in another venv (e.g. ACE-Step).

Usage:
    python scripts/steal_flash_attn.py ../ACE-Step-1.5

    # Or specify both paths explicitly:
    python scripts/steal_flash_attn.py --src /path/to/ACE-Step-1.5/.venv --dst .venv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _find_site_packages(venv: Path) -> Path | None:
    """Locate the site-packages directory inside a venv."""
    candidates = list(venv.glob("lib/python*/site-packages"))
    if not candidates:
        candidates = list(venv.glob("Lib/site-packages"))  # Windows
    return candidates[0] if candidates else None


def _find_venv(project_dir: Path) -> Path | None:
    """Find a venv directory inside a project."""
    for name in (".venv", "venv", ".env", "env"):
        candidate = project_dir / name
        if (candidate / "pyvenv.cfg").is_file():
            return candidate
    return None


def steal(src_project: Path, dst_venv: Path | None = None) -> None:
    """Copy flash-attn files from src project's venv to dst venv."""
    # Resolve source venv
    if (src_project / "pyvenv.cfg").is_file():
        src_venv = src_project  # user passed the venv directly
    else:
        src_venv = _find_venv(src_project)
        if src_venv is None:
            print(f"[ERROR] No venv found in {src_project}")
            sys.exit(1)

    src_sp = _find_site_packages(src_venv)
    if src_sp is None:
        print(f"[ERROR] No site-packages in {src_venv}")
        sys.exit(1)

    # Resolve destination venv
    if dst_venv is None:
        dst_venv = _find_venv(Path.cwd())
        if dst_venv is None:
            print("[ERROR] No venv found in current directory. Activate your venv first.")
            sys.exit(1)

    dst_sp = _find_site_packages(dst_venv)
    if dst_sp is None:
        print(f"[ERROR] No site-packages in {dst_venv}")
        sys.exit(1)

    # Check source has flash-attn
    src_so_candidates = list(src_sp.glob("flash_attn_2_cuda*.so")) + list(src_sp.glob("flash_attn_2_cuda*.pyd"))
    src_pkg = src_sp / "flash_attn"
    src_dist = list(src_sp.glob("flash_attn-*.dist-info"))
    src_hopper = src_sp / "hopper"

    if not src_so_candidates or not src_pkg.is_dir():
        print(f"[ERROR] flash-attn not found in {src_sp}")
        print("  Make sure flash-attn is installed in the source venv.")
        sys.exit(1)

    src_so = src_so_candidates[0]
    size_mb = src_so.stat().st_size / (1024 * 1024)

    # Check Python version compatibility
    src_py = src_sp.parent.name  # e.g. "python3.11"
    dst_py = dst_sp.parent.name
    if src_py != dst_py:
        print(f"[WARN] Python version mismatch: source={src_py}, dest={dst_py}")
        print("  The compiled .so may not be compatible.")
        resp = input("  Continue anyway? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            sys.exit(0)

    # Check if already installed in dest
    dst_so = dst_sp / src_so.name
    if dst_so.exists():
        print(f"[INFO] flash-attn already exists in destination: {dst_so}")
        resp = input("  Overwrite? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("  Skipped.")
            return

    # Copy files
    print(f"\n  Source: {src_sp}")
    print(f"  Dest:   {dst_sp}\n")

    # 1. The big .so file
    print(f"  Copying {src_so.name} ({size_mb:.0f} MB)...", end=" ", flush=True)
    shutil.copy2(src_so, dst_sp / src_so.name)
    print("OK")

    # 2. The flash_attn Python package
    dst_pkg = dst_sp / "flash_attn"
    if dst_pkg.exists():
        shutil.rmtree(dst_pkg)
    print(f"  Copying flash_attn/ package...", end=" ", flush=True)
    shutil.copytree(src_pkg, dst_pkg)
    print("OK")

    # 3. The dist-info metadata
    for dist in src_dist:
        dst_dist = dst_sp / dist.name
        if dst_dist.exists():
            shutil.rmtree(dst_dist)
        print(f"  Copying {dist.name}/...", end=" ", flush=True)
        shutil.copytree(dist, dst_dist)
        print("OK")

    # 4. The hopper directory (if present)
    if src_hopper.is_dir():
        dst_hopper = dst_sp / "hopper"
        if dst_hopper.exists():
            shutil.rmtree(dst_hopper)
        print(f"  Copying hopper/ ...", end=" ", flush=True)
        shutil.copytree(src_hopper, dst_hopper)
        print("OK")

    print(f"\n[OK] flash-attn copied successfully!")
    print(f"  Verify with: python -c \"import flash_attn; print(flash_attn.__version__)\"")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy compiled flash-attn from another project's venv",
    )
    parser.add_argument(
        "source",
        help="Path to the source project (e.g. ../ACE-Step-1.5) or its venv",
    )
    parser.add_argument(
        "--dst",
        default=None,
        help="Path to destination venv (default: auto-detect in CWD)",
    )
    args = parser.parse_args()

    src = Path(args.source).resolve()
    dst = Path(args.dst).resolve() if args.dst else None
    steal(src, dst)


if __name__ == "__main__":
    main()
