# install_requirements.py
"""
Install all dependencies for the CarRacing RL project into the *current* Python
environment (ideally a .venv in the project folder).

Usage (Windows, PowerShell):

    python -m venv .venv
    .venv\\Scripts\\Activate.ps1
    python install_requirements.py

This script uses `sys.executable -m pip install ...` so whatever interpreter
you run it with is where the packages get installed.
"""

import subprocess
import sys


# --- PACKAGE LISTS ----------------------------------------------------------
# Option A: Original deps (like in your Colab scripts)
# NOTE: On Windows with modern Python, `gymnasium[box2d]` pulls `box2d-py`
# which often fails to build. If that happens, use Option B below instead.

BASE_PACKAGES = [
    # Core RL / env stack
    "swig",
    "gymnasium[box2d]",   # may fail on Windows; see note below
    "pygame",

    # DL stack
    "torch",
    "torchvision",

    # Numerics / data / plotting
    "numpy",
    "pandas",
    "matplotlib",

    # Image processing
    "opencv-python",

    # Config handling
    "pyyaml",
]

# Option B: If gymnasium[box2d] keeps failing on Windows, comment out the
# "gymnasium[box2d]" line above and use these two instead:
#
#   "gymnasium",
#   "box2d==2.3.10",
#
# That replaces the broken box2d-py with a maintained fork.


# --- INSTALL HELPER ---------------------------------------------------------


def pip_install(package: str) -> None:
    """Run `python -m pip install <package>` with the current interpreter."""
    print(f"\n=== Installing {package} ===")
    cmd = [sys.executable, "-m", "pip", "install", package]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install {package}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Exit code: {e.returncode}")
        # Don't abort the whole script; just continue to next package.


def main() -> None:
    print(f"Using Python interpreter: {sys.executable}")
    print("Installing CarRacing RL dependencies...\n")

    for pkg in BASE_PACKAGES:
        pip_install(pkg)

    print("\nAll install commands finished (check output above for any errors).")


if __name__ == "__main__":
    main()
