#!/usr/bin/env python3

import argparse
import importlib
import platform
import subprocess
import sys
from typing import Dict, Tuple, List

PackageSpec = Tuple[str, str]  # (pip_name, import_name)

def tf_pip_name() -> str:
    # Apple Silicon (arm64) often uses tensorflow-macos
    if platform.system() == "Darwin" and platform.machine() in {"arm64", "x86_64"}:
        # On Intel Macs, standard tensorflow wheels may also apply, but tensorflow-macos works only for arm64.
        return "tensorflow-macos" if platform.machine() == "arm64" else "tensorflow"
    return "tensorflow"

BASE_PACKAGES: Dict[str, PackageSpec] = {
    # pip name                  import name
    "pandas":                 ("pandas", "pandas"),
    "numpy":                  ("numpy", "numpy"),
    "matplotlib":             ("matplotlib", "matplotlib"),
    "seaborn":                ("seaborn", "seaborn"),
    "scipy":                  ("scipy", "scipy"),
    "statsmodels":            ("statsmodels", "statsmodels"),
    "scikit-learn":           ("scikit-learn", "sklearn"),
    "tensorflow":             (tf_pip_name(), "tensorflow"),
    "kagglehub":              ("kagglehub", "kagglehub"),
}

EXTRA_PACKAGES: Dict[str, PackageSpec] = {
    "plotly":                 ("plotly", "plotly"),
    "xgboost":                ("xgboost", "xgboost"),
    "lightgbm":               ("lightgbm", "lightgbm"),
    "catboost":               ("catboost", "catboost"),
    "imbalanced-learn":       ("imbalanced-learn", "imblearn"),
    "polars":                 ("polars", "polars"),
    "pyarrow":                ("pyarrow", "pyarrow"),
    "optuna":                 ("optuna", "optuna"),
    "shap":                   ("shap", "shap"),
    "pmdarima":               ("pmdarima", "pmdarima"),
}

def is_installed(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except Exception:
        return False

def pip_install(pip_name: str, upgrade: bool = False, quiet: bool = False) -> int:
    args = [sys.executable, "-m", "pip", "install", pip_name]
    if upgrade:
        args.append("--upgrade")
    if quiet:
        args.append("--quiet")
    print(f"→ Installing: {pip_name}")
    return subprocess.call(args)

def install_group(group: Dict[str, PackageSpec], upgrade: bool, quiet: bool, dry_run: bool) -> Tuple[List[str], List[str]]:
    installed_ok, failed = [], []
    for display_name, (pip_name, import_name) in group.items():
        if is_installed(import_name):
            print(f"✓ Already installed: {display_name} (import '{import_name}')")
            continue
        if dry_run:
            print(f"[DRY-RUN] Would install: {display_name} via '{pip_name}'")
            continue
        code = pip_install(pip_name, upgrade=upgrade, quiet=quiet)
        if code == 0 and is_installed(import_name):
            print(f"✓ Installed: {display_name}")
            installed_ok.append(display_name)
        else:
            print(f"✗ Failed: {display_name} (pip: {pip_name})")
            failed.append(display_name)
    return installed_ok, failed

def parse_args():
    p = argparse.ArgumentParser(description="Install missing data-science packages.")
    p.add_argument("--extras", action="store_true", help="Also install optional/extra packages.")
    p.add_argument("--only-extras", action="store_true", help="Install ONLY extras (skip base).")
    p.add_argument("--upgrade", action="store_true", help="Use pip --upgrade when installing.")
    p.add_argument("--quiet", action="store_true", help="Pass --quiet to pip.")
    p.add_argument("--dry-run", action="store_true", help="Show what would be installed, but don't install.")
    return p.parse_args()

def main():
    args = parse_args()
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print()

    total_installed, total_failed = [], []

    if not args.only_extras:
        print("=== BASE PACKAGES ===")
        ok, bad = install_group(BASE_PACKAGES, upgrade=args.upgrade, quiet=args.quiet, dry_run=args.dry_run)
        total_installed += ok
        total_failed += bad
        print()

    if args.extras or args.only_extras:
        print("=== EXTRA PACKAGES ===")
        ok, bad = install_group(EXTRA_PACKAGES, upgrade=args.upgrade, quiet=args.quiet, dry_run=args.dry_run)
        total_installed += ok
        total_failed += bad
        print()

    print("=== SUMMARY ===")
    if total_installed:
        print("Installed:", ", ".join(total_installed))
    else:
        print("Installed: (none)")
    if total_failed:
        print("Failed:", ", ".join(total_failed))
    else:
        print("Failed: (none)")
    print("\nTip: re-open your terminal/IDE or restart your Python kernel if imports are not recognized immediately.")

if __name__ == "__main__":
    main()
